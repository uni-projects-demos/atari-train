use crate::{
    config::GameSpec,
    env,
    model::{BatchBuffer, DQN},
    state::State,
};
use anyhow::Result;
use burn::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub seed: u64,
    pub steps: u64,
    pub raw_return: f32,
    #[serde(alias = "terminated")]
    pub is_terminated: bool,
    #[serde(alias = "truncated")]
    pub is_truncated: bool,
    #[serde(alias = "evaluation_cutoff")]
    pub is_eval_cutoff: bool,
    pub scores: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Eval {
    pub game: String,
    pub episodes: Vec<Episode>,
    pub completed: usize,
    pub mean_completed_return: Option<f32>,
    #[serde(alias = "wall_seconds")]
    pub wall_secs: f64,
}

pub fn eval(
    model: &DQN,
    spec: &GameSpec,
    seed: u64,
    episodes: usize,
    max_steps: u64,
) -> Result<Eval> {
    let start: Instant = Instant::now();
    let model: DQN = model.valid();
    let dev: Device = model.devices()[0].clone();
    let mut envs: Vec<Box<dyn env::Env>> = (0..episodes)
        .map(|idx: usize| {
            let offset: u64 = u64::try_from(idx).expect("episode index fits u64");
            env::create(spec, seed.wrapping_add(offset))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut states: Vec<State> = envs
        .iter_mut()
        .map(|env: &mut Box<dyn env::Env>| env.reset())
        .collect::<Result<Vec<_>>>()?;
    let mut recs: Vec<Episode> = (0..episodes)
        .map(|idx: usize| Episode {
            seed: seed.wrapping_add(u64::try_from(idx).expect("episode index fits u64")),
            steps: 0,
            raw_return: 0.0,
            is_terminated: false,
            is_truncated: false,
            is_eval_cutoff: false,
            scores: None,
        })
        .collect();
    let mut is_active: Vec<bool> = vec![true; episodes];
    let mut buf: BatchBuffer = BatchBuffer::default();

    while is_active.iter().any(|&is_active: &bool| is_active) {
        let ids: Vec<usize> = is_active
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_active): (usize, &bool)| is_active.then_some(idx))
            .collect();
        let inp: Tensor<4> = buf.tensor(
            ids.iter().map(|&idx: &usize| &states[idx]),
            spec.shape(),
            &dev,
        );
        let vals: Vec<i64> = model
            .forward(inp)
            .argmax(1)
            .reshape([ids.len()])
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let mut actions: Vec<usize> = vec![0; episodes];
        for (batch_idx, &env_idx) in ids.iter().enumerate() {
            actions[env_idx] = usize::try_from(vals[batch_idx])
                .map_err(|_| anyhow::anyhow!("Negative policy action index"))?;
        }

        let results: Vec<Result<(usize, env::Step)>> = envs
            .par_iter_mut()
            .enumerate()
            .filter(|(idx, _)| is_active[*idx])
            .map(|(idx, env)| Ok((idx, env.step(actions[idx])?)))
            .collect();
        for item in results {
            let (idx, step): (usize, env::Step) = item?;
            let rec: &mut Episode = &mut recs[idx];
            rec.steps += 1;
            rec.raw_return += step.reward;
            rec.scores = step.scores;
            rec.is_terminated = step.is_terminated;
            rec.is_eval_cutoff = !step.is_done() && rec.steps >= max_steps;
            rec.is_truncated = step.is_truncated || rec.is_eval_cutoff;
            is_active[idx] = !rec.is_terminated && !rec.is_truncated;
            states[idx] = step.state;
        }
    }

    let completed: usize = recs
        .iter()
        .filter(|rec: &&Episode| !rec.is_eval_cutoff && (rec.is_terminated || rec.is_truncated))
        .count();
    let mean_completed_return: Option<f32> = (completed > 0).then(|| {
        recs.iter()
            .filter(|rec: &&Episode| !rec.is_eval_cutoff && (rec.is_terminated || rec.is_truncated))
            .map(|rec: &Episode| rec.raw_return)
            .sum::<f32>()
            / completed as f32
    });

    Ok(Eval {
        game: spec.key.clone(),
        episodes: recs,
        completed,
        mean_completed_return,
        wall_secs: start.elapsed().as_secs_f64(),
    })
}
