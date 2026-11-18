use crate::{config::GameSpec, state::State};
use anyhow::Result;

#[cfg(feature = "native")]
pub mod atari;

#[derive(Debug)]
pub struct Step {
    pub state: State,
    pub reward: f32,
    pub is_terminated: bool,
    pub is_truncated: bool,
    pub is_life_lost: bool,
    pub scores: Option<[f32; 2]>,
}

impl Step {
    pub const fn is_done(&self) -> bool {
        self.is_terminated || self.is_truncated
    }
}

pub trait Env: Send {
    fn reset(&mut self) -> Result<State>;
    fn step(&mut self, action: usize) -> Result<Step>;
}

pub fn create(spec: &GameSpec, seed: u64) -> Result<Box<dyn Env>> {
    spec.validate()?;
    #[cfg(feature = "native")]
    {
        Ok(Box::new(atari::AtariEnv::new(spec.clone(), seed)?))
    }
    #[cfg(not(feature = "native"))]
    {
        let _ = (spec, seed);
        anyhow::bail!("This build has no native ALE; rebuild with the default native feature")
    }
}
