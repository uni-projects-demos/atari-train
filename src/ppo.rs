use crate::{
    checkpoint::{self, Metadata},
    config::{Alg, RunConfig},
    env,
    eval::{self, Eval},
    model::{BatchBuffer, DQN},
    onnx,
    state::State,
    train::TrainingResult,
};
use anyhow::{Result, ensure};
use burn::{
    optim::{AdamConfig, GradientsParams, ModuleOptimizer, grad_clipping::GradientClippingConfig},
    prelude::*,
    tensor::activation::{log_softmax, softmax},
};
use rand::{RngExt, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

struct Worker {
    env: Box<dyn env::Env>,
    state: State,
    ep_return: f32,
    ep_steps: u64,
}

struct Step {
    state: State,
    action: usize,
    reward: f32,
    is_done: bool,
    old_logp: f32,
    val: f32,
    adv: f32,
    ret: f32,
}

pub fn train(
    run_cfg: RunConfig,
    dev: Device,
    out: &Path,
    init: Option<DQN>,
) -> Result<TrainingResult> {
    run_cfg.validate()?;
    ensure!(run_cfg.training.alg == Alg::Ppo, "PPO run requires alg=ppo");
    ensure!(
        !out.exists() || std::fs::read_dir(out)?.next().is_none(),
        "Run directory is not empty: {}",
        out.display()
    );
    std::fs::create_dir_all(out)?;
    run_cfg.save(&out.join("config.json"))?;
    let cfg = &run_cfg.training;
    let spec = &run_cfg.game;
    let start = Instant::now();
    let dev = dev.autodiff();
    dev.seed(cfg.seed);
    let mut model = init
        .map(|m| m.train().fork(&dev))
        .unwrap_or_else(|| DQN::new_algo(spec, Alg::Ppo, cfg.is_dueling, &dev));
    let mut opt: ModuleOptimizer = AdamConfig::new()
        .with_epsilon(1e-5)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0x3dda_84a8);
    let mut workers: Vec<Worker> = (0..cfg.envs)
        .map(|idx| {
            let mut env = env::create(spec, cfg.seed.wrapping_add(idx as u64))?;
            let state = env.reset()?;
            Ok(Worker {
                env,
                state,
                ep_return: 0.0,
                ep_steps: 0,
            })
        })
        .collect::<Result<_>>()?;
    let mut episode_log = BufWriter::new(File::create(out.join("episodes.csv"))?);
    writeln!(
        episode_log,
        "decision,env,episode,steps,raw_return,is_terminated,is_truncated,model_score,game_score"
    )?;
    let mut train_log = BufWriter::new(File::create(out.join("training.csv"))?);
    writeln!(
        train_log,
        "decision,updates,loss,policy_loss,val_loss,entropy,decisions_per_sec"
    )?;
    let mut eval_log = BufWriter::new(File::create(out.join("evals.jsonl"))?);
    let mut buf = BatchBuffer::default();
    let mut decisions = 0_u64;
    let mut updates = 0_u64;
    let mut episodes = 0_u64;
    let mut next_eval = cfg.eval_every;
    let mut best = f32::NEG_INFINITY;
    let mut final_loss = None;
    let mut latest: Option<Eval> = None;

    while decisions < cfg.steps {
        let mut rollouts: Vec<Vec<Step>> = (0..cfg.envs)
            .map(|_| Vec::with_capacity(cfg.ppo_rollout))
            .collect();
        for _ in 0..cfg.ppo_rollout {
            if decisions >= cfg.steps {
                break;
            }
            let n = (cfg.steps - decisions).min(cfg.envs as u64) as usize;
            let valid = model.valid();
            let raw_dev = valid.devices()[0].clone();
            let inp = buf.tensor(
                workers[..n].iter().map(|w| &w.state),
                spec.shape(),
                &raw_dev,
            );
            let probs: Vec<f32> = softmax(valid.forward(inp.clone()), 1)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
            let vals: Vec<f32> = valid
                .val(inp)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
            let actions = spec.actions.len();
            for idx in 0..n {
                let row = &probs[idx * actions..(idx + 1) * actions];
                let threshold = rng.random::<f32>();
                let mut cumulative = 0.0;
                let mut action = actions - 1;
                for (a, &p) in row.iter().enumerate() {
                    cumulative += p;
                    if threshold <= cumulative {
                        action = a;
                        break;
                    }
                }
                let w = &mut workers[idx];
                let state = w.state.clone();
                let step = w.env.step(action)?;
                let is_done = step.is_done();
                let is_episode_end = is_done || step.is_life_lost;
                w.ep_return += step.reward;
                w.ep_steps += 1;
                rollouts[idx].push(Step {
                    state,
                    action,
                    reward: if cfg.is_clip_reward {
                        step.reward.clamp(-1.0, 1.0)
                    } else {
                        step.reward
                    },
                    is_done: is_episode_end,
                    old_logp: row[action].max(1e-10).ln(),
                    val: vals[idx],
                    adv: 0.0,
                    ret: 0.0,
                });
                if is_done {
                    episodes += 1;
                    let scores = step
                        .scores
                        .map(|s| [s[0].to_string(), s[1].to_string()])
                        .unwrap_or_default();
                    writeln!(
                        episode_log,
                        "{},{idx},{episodes},{},{},{},{},{},{}",
                        decisions + idx as u64 + 1,
                        w.ep_steps,
                        w.ep_return,
                        step.is_terminated,
                        step.is_truncated,
                        scores[0],
                        scores[1]
                    )?;
                    w.ep_return = 0.0;
                    w.ep_steps = 0;
                    w.state = w.env.reset()?;
                } else {
                    w.state = step.state;
                }
            }
            decisions += n as u64;
        }

        let valid = model.valid();
        let raw_dev = valid.devices()[0].clone();
        let bootstrap: Vec<f32> = valid
            .val(buf.tensor(workers.iter().map(|w| &w.state), spec.shape(), &raw_dev))
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        for (idx, rollout) in rollouts.iter_mut().enumerate() {
            let mut next_val = bootstrap[idx];
            let mut gae = 0.0_f32;
            for tran in rollout.iter_mut().rev() {
                let alive = if tran.is_done { 0.0 } else { 1.0 };
                let delta = tran.reward + cfg.gamma * next_val * alive - tran.val;
                gae = delta + cfg.gamma * cfg.ppo_gae_lambda * alive * gae;
                tran.adv = gae;
                tran.ret = gae + tran.val;
                next_val = tran.val;
            }
        }
        let steps: Vec<Step> = rollouts.into_iter().flatten().collect();
        let count = steps.len();
        ensure!(count > 0, "Empty PPO rollout");
        let mean = steps.iter().map(|s| f64::from(s.adv)).sum::<f64>() / count as f64;
        let var = steps
            .iter()
            .map(|s| (f64::from(s.adv) - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        let std = (var + 1e-8).sqrt();
        let mut idxs: Vec<usize> = (0..count).collect();
        for _ in 0..cfg.ppo_epochs {
            idxs.shuffle(&mut rng);
            for chunk in idxs.chunks(cfg.ppo_minibatch) {
                let n = chunk.len();
                let inp = buf.tensor(chunk.iter().map(|&i| &steps[i].state), spec.shape(), &dev);
                let acts: Tensor<2, Int> = Tensor::from_data(
                    TensorData::new(
                        chunk
                            .iter()
                            .map(|&i| steps[i].action as i64)
                            .collect::<Vec<_>>(),
                        [n, 1],
                    ),
                    &dev,
                );
                let old_logps: Tensor<1> = Tensor::from_data(
                    TensorData::new(
                        chunk.iter().map(|&i| steps[i].old_logp).collect::<Vec<_>>(),
                        [n],
                    ),
                    &dev,
                );
                let advs: Tensor<1> = Tensor::from_data(
                    TensorData::new(
                        chunk
                            .iter()
                            .map(|&i| ((f64::from(steps[i].adv) - mean) / std) as f32)
                            .collect::<Vec<_>>(),
                        [n],
                    ),
                    &dev,
                );
                let returns: Tensor<1> = Tensor::from_data(
                    TensorData::new(chunk.iter().map(|&i| steps[i].ret).collect::<Vec<_>>(), [n]),
                    &dev,
                );
                let logits = model.forward(inp.clone());
                let logp_all = log_softmax(logits, 1);
                let logp: Tensor<1> = logp_all.clone().gather(1, acts).reshape([n]);
                let ratio = (logp - old_logps).exp();
                let s1 = ratio.clone() * advs.clone();
                let s2 = ratio.clamp(1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * advs;
                let min_surr = (s1.clone() + s2.clone() - (s1 - s2).abs()) * 0.5;
                let policy_loss = min_surr.mean() * -1.0;
                let val_loss = (model.val(inp) - returns).powf_scalar(2.0).mean();
                let entropy = (softmax(logp_all.clone(), 1) * logp_all).sum_dim(1).mean() * -1.0;
                let loss = policy_loss.clone() + val_loss.clone() * cfg.ppo_val_coef
                    - entropy.clone() * cfg.ppo_entropy_coef;
                let loss_val = loss.clone().into_scalar::<f32>();
                ensure!(loss_val.is_finite(), "Nonfinite PPO loss");
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = opt.step(cfg.lr, model, grads);
                updates += 1;
                final_loss = Some(loss_val);
                if updates.is_multiple_of(100) || updates == 1 {
                    let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
                    writeln!(
                        train_log,
                        "{decisions},{updates},{loss_val},{},{},{},{}",
                        policy_loss.into_scalar::<f32>(),
                        val_loss.into_scalar::<f32>(),
                        entropy.into_scalar::<f32>(),
                        decisions as f64 / elapsed
                    )?;
                }
            }
        }
        if decisions >= next_eval || decisions == cfg.steps {
            let report = eval::eval(
                &model,
                spec,
                cfg.seed.wrapping_add(1_000_000),
                cfg.eval_episodes,
                cfg.eval_max_steps,
            )?;
            let meta = Metadata {
                training_step: decisions,
                ..Metadata::fresh(run_cfg.clone())
            };
            checkpoint::save(&out.join("last.safetensors"), &model, &meta)?;
            if report.completed == cfg.eval_episodes
                && let Some(score) = report.mean_completed_return
                && score > best
            {
                best = score;
                checkpoint::save(&out.join("best.safetensors"), &model, &meta)?;
            }
            writeln!(
                eval_log,
                "{}",
                json!({"decisions": decisions, "updates": updates, "evaluation": report})
            )?;
            eval_log.flush()?;
            episode_log.flush()?;
            train_log.flush()?;
            println!(
                "{} algorithm=ppo decisions={} updates={} completed_eval={}/{} mean_completed={:?}",
                spec.key,
                decisions,
                updates,
                report.completed,
                cfg.eval_episodes,
                report.mean_completed_return
            );
            latest = Some(report);
            while next_eval <= decisions {
                next_eval = next_eval.saturating_add(cfg.eval_every);
            }
        }
    }
    let exported_checkpoint = if best.is_finite() {
        "best.safetensors"
    } else {
        "last.safetensors"
    };
    let (meta, export_model) = checkpoint::load(&out.join(exported_checkpoint), &Device::flex())?;
    onnx::export(&out.join("policy.onnx"), &export_model, &meta, false)?;
    let secs = start.elapsed().as_secs_f64().max(f64::EPSILON);
    let result = TrainingResult {
        game: spec.key.clone(),
        decisions,
        updates,
        episodes,
        wall_secs: secs,
        decisions_per_sec: decisions as f64 / secs,
        final_loss,
        exported_checkpoint: exported_checkpoint.to_owned(),
        eval: latest.expect("Final evaluation is mandatory"),
    };
    std::fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
