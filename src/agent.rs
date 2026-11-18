use crate::{
    config::{Alg, RunConfig, TrainingConfig},
    model::{BatchBuffer, DQN, RAINBOW_ATOMS, RAINBOW_MAX, RAINBOW_MIN},
    replay::Sample,
    state::State,
};
use anyhow::Result;
use burn::{
    optim::{AdamConfig, GradientsParams, ModuleOptimizer, grad_clipping::GradientClippingConfig},
    prelude::*,
    tensor::activation::{log_softmax, softmax},
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct Agent {
    pub model: DQN,
    pub target: DQN,
    opt: ModuleOptimizer,
    pub updates: u64,
    dev: Device,
    cfg: RunConfig,
    buf: BatchBuffer,
}

impl Agent {
    pub fn new(cfg: RunConfig, dev: Device, init: Option<DQN>) -> Self {
        let dev: Device = dev.autodiff();
        dev.seed(cfg.training.seed);
        let model: DQN = init
            .map(|model: DQN| model.train().fork(&dev))
            .unwrap_or_else(|| {
                DQN::new_algo(&cfg.game, cfg.training.alg, cfg.training.is_dueling, &dev)
            });
        let target: DQN = model.valid();
        let opt: ModuleOptimizer = AdamConfig::new()
            .with_epsilon(1.5e-4)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)))
            .init();
        Self {
            model,
            target,
            opt,
            updates: 0,
            dev,
            cfg,
            buf: BatchBuffer::default(),
        }
    }

    pub fn actions(&mut self, states: &[State]) -> Result<Vec<usize>> {
        let valid: DQN = self.model.valid();
        let inp: Tensor<4> = self
            .buf
            .tensor(states, self.cfg.game.shape(), &valid.devices()[0]);
        let out: Vec<i64> = valid
            .forward(inp)
            .argmax(1)
            .reshape([states.len()])
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        out.into_iter()
            .map(|val: i64| {
                usize::try_from(val).map_err(|_| anyhow::anyhow!("Negative policy action index"))
            })
            .collect()
    }

    pub fn update(mut self, sample: &Sample) -> Result<(Self, f32, Vec<f32>)> {
        match self.cfg.training.alg {
            Alg::Rainbow => return self.update_rainbow(sample),
            Alg::Iqn => return self.update_iqn(sample),
            Alg::Ppo => anyhow::bail!("PPO uses on-policy rollouts, not replay updates"),
            Alg::Dqn => {}
        }
        let n: usize = sample.trans.len();
        let shape: [usize; 3] = self.cfg.game.shape();
        let dev: &Device = &self.dev;
        let inp: Tensor<4> = self
            .buf
            .tensor(sample.trans.iter().map(|t| &t.state), shape, dev);
        let action_vals: Vec<i64> = sample
            .trans
            .iter()
            .map(|t| i64::try_from(t.action).expect("policy action index fits i64"))
            .collect();
        let acts: Tensor<2, Int> =
            Tensor::<2, Int>::from_data(TensorData::new(action_vals, [n, 1]), dev);
        let q: Tensor<1> = self.model.forward(inp).gather(1, acts).reshape([n]);

        let valid: DQN = self.model.valid();
        let raw_dev: Device = valid.devices()[0].clone();
        let next: Tensor<4> =
            self.buf
                .tensor(sample.trans.iter().map(|t| &t.next), shape, &raw_dev);
        let best = valid.forward(next.clone()).argmax(1);
        let next_q: Tensor<1> = self.target.forward(next).gather(1, best).reshape([n]);
        let rewards: Tensor<1> = Tensor::<1>::from_data(
            TensorData::new(
                sample.trans.iter().map(|t| t.reward).collect::<Vec<f32>>(),
                [n],
            ),
            &raw_dev,
        );
        let discounts: Tensor<1> = Tensor::<1>::from_data(
            TensorData::new(
                sample
                    .trans
                    .iter()
                    .map(|t| t.discount)
                    .collect::<Vec<f32>>(),
                [n],
            ),
            &raw_dev,
        );
        let targets: Tensor<1> = (rewards + discounts * next_q).to_device(dev).detach();
        let diff: Tensor<1> = q - targets;
        let abs: Tensor<1> = diff.clone().abs();
        let quad: Tensor<1> = abs.clone().clamp_max(1.0);
        let lin: Tensor<1> = abs.clone() - quad.clone();
        let huber: Tensor<1> = quad.powf_scalar(2.0) * 0.5 + lin;
        let wts: Tensor<1> =
            Tensor::<1>::from_data(TensorData::new(sample.weights.clone(), [n]), dev);
        let loss: Tensor<1> = (huber * wts).mean();
        let val: f32 = loss.clone().into_scalar::<f32>();
        let errs: Vec<f32> = abs
            .detach()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        anyhow::ensure!(
            val.is_finite() && errs.iter().all(|err: &f32| err.is_finite()),
            "Nonfinite TD loss; inspect learning-rate/reward settings"
        );
        let grads: GradientsParams = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self.opt.step(self.cfg.training.lr, self.model, grads);
        self.updates += 1;
        if self.updates.is_multiple_of(self.cfg.training.target_every) {
            self.target = self.model.valid();
        }
        Ok((self, val, errs))
    }

    fn finish(mut self, loss: Tensor<1>, priorities: Vec<f32>) -> Result<(Self, f32, Vec<f32>)> {
        let val = loss.clone().into_scalar::<f32>();
        anyhow::ensure!(
            val.is_finite() && priorities.iter().all(|p| p.is_finite()),
            "Nonfinite loss/priority"
        );
        let grads: GradientsParams = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self.opt.step(self.cfg.training.lr, self.model, grads);
        self.updates += 1;
        if self.updates.is_multiple_of(self.cfg.training.target_every) {
            self.target = self.model.valid();
        }
        Ok((self, val, priorities))
    }

    fn update_rainbow(mut self, sample: &Sample) -> Result<(Self, f32, Vec<f32>)> {
        let n = sample.trans.len();
        let actions = self.cfg.game.actions.len();
        let shape = self.cfg.game.shape();
        let valid = self.model.valid();
        let raw_dev = valid.devices()[0].clone();
        let next: Tensor<4> =
            self.buf
                .tensor(sample.trans.iter().map(|t| &t.next), shape, &raw_dev);
        let best: Vec<i64> = valid
            .forward(next.clone())
            .argmax(1)
            .reshape([n])
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let probs: Vec<f32> = softmax(self.target.rainbow_logits(next), 2)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let delta = (RAINBOW_MAX - RAINBOW_MIN) / (RAINBOW_ATOMS - 1) as f32;
        let mut projected = vec![0.0_f32; n * RAINBOW_ATOMS];
        for (i, t) in sample.trans.iter().enumerate() {
            let action = usize::try_from(best[i])?;
            anyhow::ensure!(action < actions, "Invalid target action");
            for atom in 0..RAINBOW_ATOMS {
                let z = RAINBOW_MIN + atom as f32 * delta;
                let tz = (t.reward + t.discount * z).clamp(RAINBOW_MIN, RAINBOW_MAX);
                let b = (tz - RAINBOW_MIN) / delta;
                let lo = (b.floor() as usize).min(RAINBOW_ATOMS - 1);
                let hi = (b.ceil() as usize).min(RAINBOW_ATOMS - 1);
                let prob = probs[(i * actions + action) * RAINBOW_ATOMS + atom];
                if lo == hi {
                    projected[i * RAINBOW_ATOMS + lo] += prob;
                } else {
                    projected[i * RAINBOW_ATOMS + lo] += prob * (hi as f32 - b);
                    projected[i * RAINBOW_ATOMS + hi] += prob * (b - lo as f32);
                }
            }
        }
        let inp: Tensor<4> =
            self.buf
                .tensor(sample.trans.iter().map(|t| &t.state), shape, &self.dev);
        let mut indices: Vec<i64> = Vec::with_capacity(n * RAINBOW_ATOMS);
        for t in &sample.trans {
            indices.extend(std::iter::repeat_n(t.action as i64, RAINBOW_ATOMS));
        }
        let acts: Tensor<3, Int> =
            Tensor::from_data(TensorData::new(indices, [n, 1, RAINBOW_ATOMS]), &self.dev);
        let selected = self
            .model
            .rainbow_logits(inp)
            .gather(1, acts)
            .reshape([n, RAINBOW_ATOMS]);
        let logp = log_softmax(selected, 1);
        let targets: Tensor<2> =
            Tensor::from_data(TensorData::new(projected, [n, RAINBOW_ATOMS]), &self.dev);
        let ce: Tensor<1> = (targets * logp).sum_dim(1).reshape([n]) * -1.0;
        let priorities: Vec<f32> = ce
            .clone()
            .detach()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let wts: Tensor<1> =
            Tensor::from_data(TensorData::new(sample.weights.clone(), [n]), &self.dev);
        self.finish((ce * wts).mean(), priorities)
    }

    fn update_iqn(mut self, sample: &Sample) -> Result<(Self, f32, Vec<f32>)> {
        let n = sample.trans.len();
        let nq = self.cfg.training.iqn_quantiles;
        let shape = self.cfg.game.shape();
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.cfg.training.seed ^ self.updates.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        let taus: Vec<f32> = (0..n * nq).map(|_| rng.random::<f32>()).collect();
        let next_taus: Vec<f32> = (0..n * nq).map(|_| rng.random::<f32>()).collect();
        let valid = self.model.valid();
        let raw_dev = valid.devices()[0].clone();
        let next: Tensor<4> =
            self.buf
                .tensor(sample.trans.iter().map(|t| &t.next), shape, &raw_dev);
        let best: Vec<i64> = valid
            .forward(next.clone())
            .argmax(1)
            .reshape([n])
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let mut next_indices = Vec::with_capacity(n * nq);
        for action in best {
            next_indices.extend(std::iter::repeat_n(action, nq));
        }
        let next_acts: Tensor<3, Int> =
            Tensor::from_data(TensorData::new(next_indices, [n, nq, 1]), &raw_dev);
        let next_q: Tensor<3> = self
            .target
            .forward_iqn(next, &next_taus, nq)
            .gather(2, next_acts)
            .reshape([n, 1, nq]);
        let rewards: Tensor<3> = Tensor::from_data(
            TensorData::new(
                sample.trans.iter().map(|t| t.reward).collect::<Vec<_>>(),
                [n, 1, 1],
            ),
            &raw_dev,
        );
        let discounts: Tensor<3> = Tensor::from_data(
            TensorData::new(
                sample.trans.iter().map(|t| t.discount).collect::<Vec<_>>(),
                [n, 1, 1],
            ),
            &raw_dev,
        );
        let target = (rewards + discounts * next_q).to_device(&self.dev).detach();
        let inp: Tensor<4> =
            self.buf
                .tensor(sample.trans.iter().map(|t| &t.state), shape, &self.dev);
        let mut indices = Vec::with_capacity(n * nq);
        for t in &sample.trans {
            indices.extend(std::iter::repeat_n(t.action as i64, nq));
        }
        let acts: Tensor<3, Int> =
            Tensor::from_data(TensorData::new(indices, [n, nq, 1]), &self.dev);
        let cur: Tensor<3> = self
            .model
            .forward_iqn(inp, &taus, nq)
            .gather(2, acts)
            .reshape([n, nq, 1]);
        let diff: Tensor<3> = target - cur;
        let abs: Tensor<3> = diff.clone().abs();
        let quad = abs.clone().clamp_max(1.0);
        let huber = quad.clone().powf_scalar(2.0) * 0.5 + (abs.clone() - quad);
        let indicator = (diff.clone().detach().sign() * -0.5) + 0.5;
        let tau: Tensor<3> = Tensor::from_data(TensorData::new(taus, [n, nq, 1]), &self.dev);
        let quantile_huber = (tau - indicator).abs() * huber;
        let per_sample: Tensor<1> = quantile_huber.mean_dims(&[1, 2]).reshape([n]);
        let priorities: Vec<f32> = abs
            .detach()
            .mean_dims(&[1, 2])
            .reshape([n])
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let wts: Tensor<1> =
            Tensor::from_data(TensorData::new(sample.weights.clone(), [n]), &self.dev);
        self.finish((per_sample * wts).mean(), priorities)
    }
}

pub fn beta(cfg: &TrainingConfig, step: u64) -> f32 {
    let warmup: u64 =
        u64::try_from(cfg.learning_starts).expect("learning_starts fits u64 on supported targets");
    let horizon: u64 = cfg.steps.saturating_sub(warmup).max(1);
    let elapsed: u64 = step.saturating_sub(warmup).min(horizon);
    let progress: f64 = elapsed as f64 / horizon as f64;
    let start: f64 = f64::from(cfg.priority_beta);
    (start + (1.0 - start) * progress) as f32
}
