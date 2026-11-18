use crate::{
    config::{Alg, GameSpec},
    state::State,
};
use anyhow::{Result, ensure};
use burn::{
    module::Param,
    nn::{
        Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::{relu, softmax},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PIXEL_SCALE: f32 = 255.0;
pub const RAINBOW_ATOMS: usize = 51;
pub const RAINBOW_MIN: f32 = -10.0;
pub const RAINBOW_MAX: f32 = 10.0;
pub const IQN_FEATURES: usize = 64;
pub const IQN_EVAL_QUANTILES: usize = 32;
pub const CONVS: [(usize, usize, usize, usize); 3] =
    [(4, 32, 8, 4), (32, 64, 4, 2), (64, 64, 3, 1)];

#[derive(Module, Debug)]
pub struct DQN {
    convs: Vec<Conv2d>,
    hidden: Linear,
    output: Linear,
    val: Option<Linear>,
    quantile: Option<Linear>,
    #[module(skip)]
    pub algo: Alg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weight {
    pub shape: Vec<usize>,
    pub vals: Vec<f32>,
}

pub type Weights = BTreeMap<String, Weight>;

impl DQN {
    pub fn new(spec: &GameSpec, is_dueling: bool, dev: &Device) -> Self {
        Self::new_algo(spec, Alg::Dqn, is_dueling, dev)
    }

    pub fn new_algo(spec: &GameSpec, algo: Alg, is_dueling: bool, dev: &Device) -> Self {
        let convs: Vec<Conv2d> = CONVS
            .iter()
            .map(
                |&(inp, out, kernel, stride): &(usize, usize, usize, usize)| {
                    Conv2dConfig::new([inp, out], [kernel, kernel])
                        .with_stride([stride, stride])
                        .init(dev)
                },
            )
            .collect();
        let hidden: Linear = LinearConfig::new(3136, spec.hidden).init(dev);
        let output_features = spec.actions.len()
            * if algo == Alg::Rainbow {
                RAINBOW_ATOMS
            } else {
                1
            };
        let output: Linear = LinearConfig::new(spec.hidden, output_features).init(dev);
        let val_features = if algo == Alg::Rainbow {
            RAINBOW_ATOMS
        } else {
            1
        };
        let val: Option<Linear> = (is_dueling || algo == Alg::Ppo)
            .then(|| LinearConfig::new(spec.hidden, val_features).init(dev));
        let quantile: Option<Linear> =
            (algo == Alg::Iqn).then(|| LinearConfig::new(IQN_FEATURES, spec.hidden).init(dev));

        Self {
            convs,
            hidden,
            output,
            val,
            quantile,
            algo,
        }
    }

    fn features(&self, inp: Tensor<4>) -> Tensor<2> {
        let mut x: Tensor<4> = inp / PIXEL_SCALE;
        for conv in &self.convs {
            x = relu(conv.forward(x));
        }
        relu(self.hidden.forward(x.flatten::<2>(1, 3)))
    }

    fn head(&self, feat: Tensor<2>) -> Tensor<2> {
        let adv: Tensor<2> = self.output.forward(feat.clone());
        if self.algo == Alg::Ppo {
            return adv;
        }
        match &self.val {
            Some(val) => {
                let mean: Tensor<2> = if self.algo == Alg::Rainbow {
                    let [n, _] = adv.dims();
                    adv.clone()
                        .reshape([n, self.n_actions(), RAINBOW_ATOMS])
                        .mean_dim(1)
                        .reshape([n, RAINBOW_ATOMS])
                } else {
                    adv.clone().mean_dim(1)
                };
                if self.algo == Alg::Rainbow {
                    let [n, _] = adv.dims();
                    ((val.forward(feat) - mean).reshape([n, 1, RAINBOW_ATOMS])
                        + adv.reshape([n, self.n_actions(), RAINBOW_ATOMS]))
                    .reshape([n, self.n_actions() * RAINBOW_ATOMS])
                } else {
                    val.forward(feat) + adv - mean
                }
            }
            None => adv,
        }
    }

    fn n_actions(&self) -> usize {
        let size = self.output.bias.as_ref().expect("linear bias").val().dims()[0];
        if self.algo == Alg::Rainbow {
            size / RAINBOW_ATOMS
        } else {
            size
        }
    }

    pub fn rainbow_logits(&self, inp: Tensor<4>) -> Tensor<3> {
        assert_eq!(self.algo, Alg::Rainbow);
        let n = inp.dims()[0];
        self.head(self.features(inp))
            .reshape([n, self.n_actions(), RAINBOW_ATOMS])
    }

    pub fn forward_iqn(&self, inp: Tensor<4>, taus: &[f32], n_quantiles: usize) -> Tensor<3> {
        assert_eq!(self.algo, Alg::Iqn);
        let n = inp.dims()[0];
        assert_eq!(taus.len(), n * n_quantiles);
        let dev: Device = inp.device();
        let mut cosines = Vec::with_capacity(taus.len() * IQN_FEATURES);
        for &tau in taus {
            for k in 1..=IQN_FEATURES {
                cosines.push((std::f32::consts::PI * k as f32 * tau).cos());
            }
        }
        let emb: Tensor<2> = Tensor::from_data(
            TensorData::new(cosines, [n * n_quantiles, IQN_FEATURES]),
            &dev,
        );
        let emb: Tensor<3> = relu(self.quantile.as_ref().expect("IQN projection").forward(emb))
            .reshape([n, n_quantiles, self.hidden.weight.val().dims()[1]]);
        let feat: Tensor<2> = self.features(inp);
        let hidden = self.hidden.weight.val().dims()[1];
        let fused: Tensor<2> =
            (feat.reshape([n, 1, hidden]) * emb).reshape([n * n_quantiles, hidden]);
        self.head(fused).reshape([n, n_quantiles, self.n_actions()])
    }

    pub fn val(&self, inp: Tensor<4>) -> Tensor<1> {
        assert_eq!(self.algo, Alg::Ppo);
        let n = inp.dims()[0];
        self.val
            .as_ref()
            .expect("PPO critic")
            .forward(self.features(inp))
            .reshape([n])
    }

    pub fn forward(&self, inp: Tensor<4>) -> Tensor<2> {
        match self.algo {
            Alg::Dqn | Alg::Ppo => self.head(self.features(inp)),
            Alg::Rainbow => {
                let dev = inp.device();
                let n = inp.dims()[0];
                let vals: Vec<f32> = (0..RAINBOW_ATOMS)
                    .map(|idx| {
                        RAINBOW_MIN
                            + idx as f32 * (RAINBOW_MAX - RAINBOW_MIN) / (RAINBOW_ATOMS - 1) as f32
                    })
                    .collect();
                let support: Tensor<3> =
                    Tensor::from_data(TensorData::new(vals, [1, 1, RAINBOW_ATOMS]), &dev);
                (softmax(self.rainbow_logits(inp), 2) * support)
                    .sum_dim(2)
                    .reshape([n, self.n_actions()])
            }
            Alg::Iqn => {
                let n = inp.dims()[0];
                let taus: Vec<f32> = (0..n)
                    .flat_map(|_| {
                        (0..IQN_EVAL_QUANTILES)
                            .map(|idx| (idx as f32 + 0.5) / IQN_EVAL_QUANTILES as f32)
                    })
                    .collect();
                self.forward_iqn(inp, &taus, IQN_EVAL_QUANTILES)
                    .mean_dim(1)
                    .reshape([n, self.n_actions()])
            }
        }
    }

    pub fn weights(&self) -> Result<Weights> {
        let mut out: Weights = Weights::new();
        for (idx, conv) in self.convs.iter().enumerate() {
            let name: String = format!("conv{}", idx + 1);
            put(&mut out, &format!("{name}.weight"), conv.weight.val())?;
            if let Some(bias) = &conv.bias {
                put(&mut out, &format!("{name}.bias"), bias.val())?;
            }
        }
        for (name, linear) in [
            ("hidden", Some(&self.hidden)),
            ("output", Some(&self.output)),
            ("value", self.val.as_ref()),
            ("quantile", self.quantile.as_ref()),
        ] {
            if let Some(layer) = linear {
                put(
                    &mut out,
                    &format!("{name}.weight"),
                    layer.weight.val().transpose(),
                )?;
                if let Some(bias) = &layer.bias {
                    put(&mut out, &format!("{name}.bias"), bias.val())?;
                }
            }
        }
        Ok(out)
    }

    pub fn load(mut self, wts: &Weights, dev: &Device) -> Result<Self> {
        let mut used: usize = 0;
        for (idx, conv) in self.convs.iter_mut().enumerate() {
            let name: String = format!("conv{}", idx + 1);
            conv.weight = Param::from_tensor(get(
                wts,
                &format!("{name}.weight"),
                conv.weight.val().dims(),
                dev,
            )?);
            used += 1;
            if let Some(bias) = &mut conv.bias {
                *bias =
                    Param::from_tensor(get(wts, &format!("{name}.bias"), bias.val().dims(), dev)?);
                used += 1;
            }
        }
        for (name, linear) in [
            ("hidden", Some(&mut self.hidden)),
            ("output", Some(&mut self.output)),
            ("value", self.val.as_mut()),
            ("quantile", self.quantile.as_mut()),
        ] {
            if let Some(layer) = linear {
                let dims: [usize; 2] = layer.weight.val().dims();
                let tensor: Tensor<2> =
                    get(wts, &format!("{name}.weight"), [dims[1], dims[0]], dev)?;
                layer.weight = Param::from_tensor(tensor.transpose());
                used += 1;
                if let Some(bias) = &mut layer.bias {
                    *bias = Param::from_tensor(get(
                        wts,
                        &format!("{name}.bias"),
                        bias.val().dims(),
                        dev,
                    )?);
                    used += 1;
                }
            }
        }
        ensure!(used == wts.len(), "Checkpoint contains unexpected tensors");
        Ok(self)
    }
}

fn put<const D: usize>(out: &mut Weights, name: &str, tensor: Tensor<D>) -> Result<()> {
    let shape: Vec<usize> = tensor.dims().to_vec();
    let vals: Vec<f32> = tensor
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("{err:?}"))?;
    ensure!(
        vals.iter().all(|val: &f32| val.is_finite()),
        "Nonfinite tensor {name}"
    );
    out.insert(name.to_owned(), Weight { shape, vals });
    Ok(())
}

fn get<const D: usize>(
    wts: &Weights,
    name: &str,
    shape: [usize; D],
    dev: &Device,
) -> Result<Tensor<D>> {
    let weight: &Weight = wts
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Missing tensor {name}"))?;
    ensure!(
        weight.shape == shape
            && weight.vals.len() == shape.iter().product::<usize>()
            && weight.vals.iter().all(|val: &f32| val.is_finite()),
        "Invalid tensor {name}"
    );
    Ok(Tensor::from_data(
        TensorData::new(weight.vals.clone(), shape),
        dev,
    ))
}

#[derive(Default)]
pub struct BatchBuffer {
    vals: Vec<f32>,
}

impl BatchBuffer {
    pub fn tensor<'a>(
        &mut self,
        states: impl IntoIterator<Item = &'a State>,
        shape: [usize; 3],
        dev: &Device,
    ) -> Tensor<4> {
        self.vals.clear();
        let elems: usize = shape.iter().product();
        let iter = states.into_iter();
        self.vals.reserve(iter.size_hint().0 * elems);
        let mut n: usize = 0;
        for state in iter {
            assert_eq!(state.elements(), elems);
            state.append(&mut self.vals);
            n += 1;
        }
        Tensor::from_data(
            TensorData::new(
                std::mem::take(&mut self.vals),
                [n, shape[0], shape[1], shape[2]],
            ),
            dev,
        )
    }
}
