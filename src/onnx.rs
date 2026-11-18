use crate::{
    checkpoint,
    config::{Alg, FRAME_HEIGHT, FRAME_STACK, FRAME_WIDTH, GameSpec, Padding},
    model::{
        CONVS, DQN, IQN_EVAL_QUANTILES, IQN_FEATURES, PIXEL_SCALE, RAINBOW_ATOMS, RAINBOW_MAX,
        RAINBOW_MIN, Weights,
    },
};
use anyhow::Result;
use prost::Message;
use std::path::Path;

const ONNX_FLOAT: i32 = 1;
const ONNX_INT64: i32 = 7;
const ATTR_INT: i32 = 2;
const ATTR_INTS: i32 = 7;

#[derive(Clone, PartialEq, Message)]
pub struct Model {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<Graph>,
    #[prost(message, repeated, tag = "8")]
    pub opset: Vec<Opset>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Opset {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, Message)]
pub struct Graph {
    #[prost(message, repeated, tag = "1")]
    pub nodes: Vec<Node>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializers: Vec<Tensor>,
    #[prost(message, repeated, tag = "11")]
    pub inputs: Vec<Val>,
    #[prost(message, repeated, tag = "12")]
    pub outputs: Vec<Val>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Node {
    #[prost(string, repeated, tag = "1")]
    pub inputs: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub outputs: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op: String,
    #[prost(message, repeated, tag = "5")]
    pub attrs: Vec<Attribute>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Attribute {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(int32, tag = "20")]
    pub kind: i32,
}

#[derive(Clone, PartialEq, Message)]
pub struct Tensor {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub dtype: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Val {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub kind: Option<Type>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Type {
    #[prost(message, optional, tag = "1")]
    pub tensor: Option<TensorType>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorType {
    #[prost(int32, tag = "1")]
    pub dtype: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<Shape>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Shape {
    #[prost(message, repeated, tag = "1")]
    pub dims: Vec<Dimension>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Dimension {
    #[prost(int64, tag = "1")]
    pub val: i64,
}

fn val(name: &str, dtype: i32, shape: &[usize]) -> Val {
    Val {
        name: name.to_owned(),
        kind: Some(Type {
            tensor: Some(TensorType {
                dtype,
                shape: Some(Shape {
                    dims: shape
                        .iter()
                        .map(|&dim: &usize| Dimension {
                            val: i64::try_from(dim).expect("ONNX dimension fits i64"),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn int(name: &str, val: i64) -> Attribute {
    Attribute {
        name: name.to_owned(),
        i: val,
        kind: ATTR_INT,
        ..Default::default()
    }
}

fn ints(name: &str, vals: &[i64]) -> Attribute {
    Attribute {
        name: name.to_owned(),
        ints: vals.to_vec(),
        kind: ATTR_INTS,
        ..Default::default()
    }
}

fn node(graph: &mut Graph, op: &str, inputs: &[&str], out: &str, attrs: Vec<Attribute>) {
    graph.nodes.push(Node {
        inputs: inputs
            .iter()
            .map(|input: &&str| (*input).to_owned())
            .collect(),
        outputs: vec![out.to_owned()],
        name: out.to_owned(),
        op: op.to_owned(),
        attrs,
    });
}

fn integer(graph: &mut Graph, name: &str, vals: &[i64]) {
    graph.initializers.push(Tensor {
        name: name.to_owned(),
        dtype: ONNX_INT64,
        dims: vec![i64::try_from(vals.len()).expect("initializer length fits i64")],
        raw: vals
            .iter()
            .flat_map(|val: &i64| val.to_le_bytes())
            .collect(),
    });
}

fn float(graph: &mut Graph, name: &str, val: f32) {
    graph.initializers.push(Tensor {
        name: name.to_owned(),
        dtype: ONNX_FLOAT,
        dims: vec![1],
        raw: val.to_le_bytes().to_vec(),
    });
}

fn floats(graph: &mut Graph, name: &str, vals: &[f32], shape: &[i64]) {
    graph.initializers.push(Tensor {
        name: name.to_owned(),
        dtype: ONNX_FLOAT,
        dims: shape.to_vec(),
        raw: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    });
}

fn dense(graph: &mut Graph, inp: &str, name: &str, out: &str) {
    node(
        graph,
        "Gemm",
        &[inp, &format!("{name}.weight"), &format!("{name}.bias")],
        out,
        vec![int("transB", 1)],
    );
}

pub fn graph(
    spec: &GameSpec,
    weights: &Weights,
    is_dueling: bool,
    algo: Alg,
    is_repair: bool,
    is_q_output: bool,
) -> Model {
    let mut graph: Graph = Graph {
        name: format!("{}-{:?}", spec.key, algo),
        ..Default::default()
    };
    for (name, weight) in weights {
        graph.initializers.push(Tensor {
            name: name.clone(),
            dims: weight
                .shape
                .iter()
                .map(|&dim: &usize| i64::try_from(dim).expect("tensor dimension fits i64"))
                .collect(),
            dtype: ONNX_FLOAT,
            raw: weight
                .vals
                .iter()
                .flat_map(|val: &f32| val.to_le_bytes())
                .collect(),
        });
    }

    let shape: [usize; 4] = [1, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH];
    graph.inputs.push(val("observation", ONNX_FLOAT, &shape));
    let mut inp: String = "observation".to_owned();

    if is_repair && spec.padding == Padding::Repeat {
        integer(&mut graph, "axis_channel", &[1]);
        integer(&mut graph, "axis_spatial", &[2, 3]);
        float(&mut graph, "zero", 0.0);
        for idx in 0..FRAME_STACK {
            let start: i64 = i64::try_from(idx).expect("frame index fits i64");
            let end: i64 = i64::try_from(idx + 1).expect("frame index fits i64");
            integer(&mut graph, &format!("start_{idx}"), &[start]);
            integer(&mut graph, &format!("end_{idx}"), &[end]);
            node(
                &mut graph,
                "Slice",
                &[
                    "observation",
                    &format!("start_{idx}"),
                    &format!("end_{idx}"),
                    "axis_channel",
                ],
                &format!("frame_{idx}"),
                Vec::new(),
            );
        }
        let mut names: Vec<String> = vec![format!("frame_{}", FRAME_STACK - 1)];
        for idx in (0..FRAME_STACK - 1).rev() {
            let frame: String = format!("frame_{idx}");
            let sum: String = format!("sum_{idx}");
            let empty: String = format!("empty_{idx}");
            let fixed: String = format!("fixed_{idx}");
            node(
                &mut graph,
                "ReduceSum",
                &[&frame, "axis_spatial"],
                &sum,
                vec![int("keepdims", 1)],
            );
            node(&mut graph, "Equal", &[&sum, "zero"], &empty, Vec::new());
            node(
                &mut graph,
                "Where",
                &[&empty, &names[0], &frame],
                &fixed,
                Vec::new(),
            );
            names.insert(0, fixed);
        }
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        node(
            &mut graph,
            "Concat",
            &name_refs,
            "adapted_history",
            vec![int("axis", 1)],
        );
        inp = "adapted_history".to_owned();
    }

    float(&mut graph, "rescale", PIXEL_SCALE);
    node(
        &mut graph,
        "Div",
        &[&inp, "rescale"],
        "normalized",
        Vec::new(),
    );
    inp = "normalized".to_owned();

    for (idx, (_, _, kernel, stride)) in CONVS.iter().enumerate() {
        let layer: usize = idx + 1;
        let conv: String = format!("conv{layer}");
        let out: String = format!("conv{layer}_out");
        let relu: String = format!("conv{layer}_relu");
        let kernel_i64: i64 = i64::try_from(*kernel).expect("kernel size fits i64");
        let stride_i64: i64 = i64::try_from(*stride).expect("stride fits i64");
        node(
            &mut graph,
            "Conv",
            &[&inp, &format!("{conv}.weight"), &format!("{conv}.bias")],
            &out,
            vec![
                ints("kernel_shape", &[kernel_i64, kernel_i64]),
                ints("strides", &[stride_i64, stride_i64]),
            ],
        );
        node(&mut graph, "Relu", &[&out], &relu, Vec::new());
        inp = relu;
    }

    node(
        &mut graph,
        "Flatten",
        &[&inp],
        "features",
        vec![int("axis", 1)],
    );
    dense(&mut graph, "features", "hidden", "hidden_out");
    node(
        &mut graph,
        "Relu",
        &["hidden_out"],
        "hidden_relu",
        Vec::new(),
    );
    match algo {
        Alg::Dqn => {
            dense(
                &mut graph,
                "hidden_relu",
                "output",
                if is_dueling { "advantage" } else { "q_values" },
            );
            if is_dueling {
                dense(&mut graph, "hidden_relu", "value", "state_value");
                node(
                    &mut graph,
                    "ReduceMean",
                    &["advantage"],
                    "advantage_mean",
                    vec![ints("axes", &[1]), int("keepdims", 1)],
                );
                node(
                    &mut graph,
                    "Sub",
                    &["advantage", "advantage_mean"],
                    "centered_advantage",
                    vec![],
                );
                node(
                    &mut graph,
                    "Add",
                    &["state_value", "centered_advantage"],
                    "q_values",
                    vec![],
                );
            }
        }
        Alg::Rainbow => {
            let actions = spec.actions.len();
            integer(
                &mut graph,
                "rainbow_shape",
                &[1, actions as i64, RAINBOW_ATOMS as i64],
            );
            dense(&mut graph, "hidden_relu", "output", "distribution_flat");
            node(
                &mut graph,
                "Reshape",
                &["distribution_flat", "rainbow_shape"],
                "advantage_atoms",
                vec![],
            );
            let logits = if is_dueling {
                dense(&mut graph, "hidden_relu", "value", "value_atoms_flat");
                integer(&mut graph, "value_shape", &[1, 1, RAINBOW_ATOMS as i64]);
                node(
                    &mut graph,
                    "Reshape",
                    &["value_atoms_flat", "value_shape"],
                    "value_atoms",
                    vec![],
                );
                node(
                    &mut graph,
                    "ReduceMean",
                    &["advantage_atoms"],
                    "mean_advantage_atoms",
                    vec![ints("axes", &[1]), int("keepdims", 1)],
                );
                node(
                    &mut graph,
                    "Sub",
                    &["advantage_atoms", "mean_advantage_atoms"],
                    "centered_atoms",
                    vec![],
                );
                node(
                    &mut graph,
                    "Add",
                    &["value_atoms", "centered_atoms"],
                    "distribution_logits",
                    vec![],
                );
                "distribution_logits"
            } else {
                "advantage_atoms"
            };
            node(
                &mut graph,
                "Softmax",
                &[logits],
                "distribution_probs",
                vec![int("axis", 2)],
            );
            let support: Vec<f32> = (0..RAINBOW_ATOMS)
                .map(|i| {
                    RAINBOW_MIN
                        + (RAINBOW_MAX - RAINBOW_MIN) * i as f32 / (RAINBOW_ATOMS - 1) as f32
                })
                .collect();
            floats(
                &mut graph,
                "rainbow_support",
                &support,
                &[1, 1, RAINBOW_ATOMS as i64],
            );
            node(
                &mut graph,
                "Mul",
                &["distribution_probs", "rainbow_support"],
                "weighted_atoms",
                vec![],
            );
            node(
                &mut graph,
                "ReduceSum",
                &["weighted_atoms"],
                "q_values",
                vec![ints("axes", &[2]), int("keepdims", 0)],
            );
        }
        Alg::Iqn => {
            let n = IQN_EVAL_QUANTILES;
            let actions = spec.actions.len();
            let mut cosines = Vec::with_capacity(n * IQN_FEATURES);
            for i in 0..n {
                let tau = (i as f32 + 0.5) / n as f32;
                for k in 1..=IQN_FEATURES {
                    cosines.push((std::f32::consts::PI * k as f32 * tau).cos());
                }
            }
            floats(
                &mut graph,
                "iqn_cosines",
                &cosines,
                &[n as i64, IQN_FEATURES as i64],
            );
            dense(&mut graph, "iqn_cosines", "quantile", "quantile_embed_raw");
            node(
                &mut graph,
                "Relu",
                &["quantile_embed_raw"],
                "quantile_embed",
                vec![],
            );
            node(
                &mut graph,
                "Mul",
                &["hidden_relu", "quantile_embed"],
                "quantile_features",
                vec![],
            );
            dense(
                &mut graph,
                "quantile_features",
                "output",
                if is_dueling {
                    "quantile_adv"
                } else {
                    "quantile_logits"
                },
            );
            let logits = if is_dueling {
                dense(&mut graph, "quantile_features", "value", "quantile_value");
                node(
                    &mut graph,
                    "ReduceMean",
                    &["quantile_adv"],
                    "quantile_adv_mean",
                    vec![ints("axes", &[1]), int("keepdims", 1)],
                );
                node(
                    &mut graph,
                    "Sub",
                    &["quantile_adv", "quantile_adv_mean"],
                    "quantile_adv_centered",
                    vec![],
                );
                node(
                    &mut graph,
                    "Add",
                    &["quantile_value", "quantile_adv_centered"],
                    "quantile_logits",
                    vec![],
                );
                "quantile_logits"
            } else {
                "quantile_logits"
            };
            integer(&mut graph, "iqn_shape", &[1, n as i64, actions as i64]);
            node(
                &mut graph,
                "Reshape",
                &[logits, "iqn_shape"],
                "all_quantiles",
                vec![],
            );
            node(
                &mut graph,
                "ReduceMean",
                &["all_quantiles"],
                "q_values",
                vec![ints("axes", &[1]), int("keepdims", 0)],
            );
        }
        Alg::Ppo => {
            dense(&mut graph, "hidden_relu", "output", "q_values");
        }
    }

    node(
        &mut graph,
        "ArgMax",
        &["q_values"],
        "policy_index",
        vec![int("axis", 1), int("keepdims", 0)],
    );
    let action_map: Vec<i64> = spec
        .actions
        .iter()
        .map(|&action: &usize| i64::try_from(action).expect("action index fits i64"))
        .collect();
    integer(&mut graph, "action_map", &action_map);
    node(
        &mut graph,
        "Gather",
        &["action_map", "policy_index"],
        "action",
        vec![int("axis", 0)],
    );
    graph.outputs.push(val("action", ONNX_INT64, &[1]));
    if is_q_output {
        graph
            .outputs
            .push(val("q_values", ONNX_FLOAT, &[1, spec.actions.len()]));
    }

    Model {
        ir_version: 8,
        producer_name: "rl-rust".to_owned(),
        graph: Some(graph),
        opset: vec![Opset {
            domain: String::new(),
            version: 17,
        }],
    }
}

pub fn export(
    path: &Path,
    model: &DQN,
    meta: &checkpoint::Metadata,
    is_q_output: bool,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    anyhow::ensure!(
        meta.config.training.alg != Alg::Ppo || !is_q_output,
        "PPO exports action logits, not Q values; omit --q-values"
    );
    let graph: Model = graph(
        &meta.config.game,
        &model.weights()?,
        meta.is_dueling,
        meta.config.training.alg,
        meta.is_repair_reset_padding,
        is_q_output,
    );
    std::fs::write(path, graph.encode_to_vec())?;

    let spec: &GameSpec = &meta.config.game;
    let sidecar: serde_json::Value = serde_json::json!({
        "game": spec.key,
        "algorithm": meta.config.training.alg,
        "arena_compatible": spec.is_arena_compatible(),
        "input": "observation",
        "output": "action",
        "shape": [1, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH],
        "actions": spec.actions,
        "padding": spec.padding,
        "grayscale_order": spec.grayscale,
        "repair_reset_padding": meta.is_repair_reset_padding,
        "normalization": "raw float32 pixels; divide by 255 inside ONNX",
        "checkpoint": meta,
        "training_performance_claim": "See measured evaluation reports; export is not evidence of game skill"
    });
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&sidecar)?,
    )?;
    Ok(())
}
