use crate::state::State;
use rand::{Rng, RngExt};
use std::collections::VecDeque;

const PRIORITY_EPS: f32 = 1e-5;

#[derive(Clone, Debug)]
pub struct Tran {
    pub state: State,
    pub action: usize,
    pub reward: f32,
    pub next: State,
    pub discount: f32,
}

#[derive(Clone)]
pub struct RawTran {
    pub state: State,
    pub action: usize,
    pub reward: f32,
    pub next: State,
    pub is_terminated: bool,
    pub is_episode_end: bool,
}

pub struct NStep {
    pending: VecDeque<RawTran>,
    n: usize,
    gamma: f32,
}

impl NStep {
    pub fn new(n: usize, gamma: f32) -> Self {
        assert!(n > 0);
        assert!(gamma.is_finite() && (0.0..=1.0).contains(&gamma));
        Self {
            pending: VecDeque::new(),
            n,
            gamma,
        }
    }

    pub fn push(&mut self, tran: RawTran) -> Vec<Tran> {
        let is_episode_end: bool = tran.is_episode_end;
        self.pending.push_back(tran);
        let mut out: Vec<Tran> = Vec::new();
        while self.pending.len() >= self.n || is_episode_end && !self.pending.is_empty() {
            let horizon: usize = self.n.min(self.pending.len());
            let first: &RawTran = &self.pending[0];
            let mut reward: f32 = 0.0;
            let mut discount: f32 = 1.0;
            let mut last: &RawTran = first;
            for item in self.pending.iter().take(horizon) {
                reward += discount * item.reward;
                discount *= self.gamma;
                last = item;
                if item.is_terminated {
                    discount = 0.0;
                    break;
                }
            }
            out.push(Tran {
                state: first.state.clone(),
                action: first.action,
                reward,
                next: last.next.clone(),
                discount,
            });
            let _ = self.pending.pop_front();
        }
        out
    }
}

pub struct Sample {
    pub trans: Vec<Tran>,
    pub idxs: Vec<usize>,
    pub weights: Vec<f32>,
}

pub struct ReplayMemory {
    entries: Vec<Option<Tran>>,
    tree: Vec<f32>,
    base: usize,
    pos: usize,
    len: usize,
    alpha: f32,
    max_prio: f32,
}

impl ReplayMemory {
    pub fn new(cap: usize, alpha: f32) -> Self {
        assert!(cap > 0);
        assert!(alpha.is_finite() && (0.0..=1.0).contains(&alpha));
        let base: usize = cap.next_power_of_two();
        Self {
            entries: (0..cap).map(|_| None).collect(),
            tree: vec![0.0; 2 * base],
            base,
            pos: 0,
            len: 0,
            alpha,
            max_prio: 1.0,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, tran: Tran) {
        self.entries[self.pos] = Some(tran);
        self.set(self.pos, self.max_prio);
        self.pos = (self.pos + 1) % self.entries.len();
        self.len = (self.len + 1).min(self.entries.len());
    }

    fn set(&mut self, idx: usize, priority: f32) {
        let mut tree_idx: usize = self.base + idx;
        self.tree[tree_idx] = priority;
        while tree_idx > 1 {
            tree_idx /= 2;
            self.tree[tree_idx] = self.tree[tree_idx * 2] + self.tree[tree_idx * 2 + 1];
        }
    }

    pub fn update(&mut self, idxs: &[usize], errs: &[f32]) {
        for (&idx, &err) in idxs.iter().zip(errs) {
            let priority: f32 = (err.abs() + PRIORITY_EPS).powf(self.alpha);
            if priority.is_finite() {
                self.max_prio = self.max_prio.max(priority);
                self.set(idx, priority);
            }
        }
    }

    pub fn sample(&self, n: usize, beta: f32, rng: &mut impl Rng) -> Sample {
        assert!(n > 0 && self.len >= n);
        assert!(beta.is_finite() && (0.0..=1.0).contains(&beta));
        let total: f32 = self.tree[1];
        assert!(total.is_finite() && total > 0.0);
        let mut sample: Sample = Sample {
            trans: Vec::with_capacity(n),
            idxs: Vec::with_capacity(n),
            weights: Vec::with_capacity(n),
        };
        for bucket in 0..n {
            let fraction: f32 = (bucket as f32 + rng.random::<f32>()) / n as f32;
            let mut mass: f32 = (fraction * total).min(total.next_down());
            let mut tree_idx: usize = 1;
            while tree_idx < self.base {
                let left: usize = tree_idx * 2;
                if mass < self.tree[left] {
                    tree_idx = left;
                } else {
                    mass -= self.tree[left];
                    tree_idx = left + 1;
                }
            }
            let idx: usize = (tree_idx - self.base).min(self.entries.len() - 1);
            let prob: f32 = self.tree[tree_idx] / total;
            sample.weights.push((self.len as f32 * prob).powf(-beta));
            sample.idxs.push(idx);
            sample.trans.push(
                self.entries[idx]
                    .as_ref()
                    .expect("positive priority has a transition")
                    .clone(),
            );
        }
        let max_weight: f32 = sample.weights.iter().copied().fold(0.0, f32::max);
        assert!(max_weight.is_finite() && max_weight > 0.0);
        for weight in &mut sample.weights {
            *weight /= max_weight;
        }
        sample
    }
}
