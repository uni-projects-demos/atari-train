mod file;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const FRAME_STACK: usize = 4;
pub const FRAME_WIDTH: usize = 84;
pub const FRAME_HEIGHT: usize = 84;
pub const ALE_ACTIONS: usize = 18;
pub const MAX_EPISODE_STEPS: u64 = 27_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Padding {
    Zero,
    Repeat,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Grayscale {
    BeforeResize,
    AfterResize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Scoring {
    Signed,
    Ram {
        #[serde(alias = "addresses")]
        addrs: [usize; 2],
        #[serde(alias = "bcd")]
        is_bcd: bool,
        overrides: Vec<[u16; 2]>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameSpec {
    pub key: String,
    pub rom: PathBuf,
    pub scoring: Scoring,
    pub actions: Vec<usize>,
    #[serde(alias = "minimal_actions")]
    pub min_actions: Vec<i32>,
    pub grayscale: Grayscale,
    pub padding: Padding,
    pub frame_skip: u32,
    #[serde(alias = "sticky_probability")]
    pub sticky_prob: f32,
    #[serde(alias = "max_pool")]
    pub is_max_pool: bool,
    pub noop_max: u32,
    pub noop_frame_skip: u32,
    pub reset_actions: Vec<usize>,
    #[serde(alias = "episodic_life")]
    pub is_episodic_life: bool,
    pub max_episode_steps: u64,
    pub hidden: usize,
}

impl GameSpec {
    pub const fn shape(&self) -> [usize; 3] {
        [FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH]
    }

    pub fn is_arena_compatible(&self) -> bool {
        GAMES.contains(&self.key.as_str())
            && Self::named(&self.key).is_ok_and(|target: Self| {
                self.min_actions == target.min_actions
                    && self.grayscale == target.grayscale
                    && (self.padding == target.padding
                        || self.padding == Padding::Repeat && target.padding == Padding::Zero)
            })
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.key.is_empty()
                && self
                    .key
                    .chars()
                    .all(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "Invalid game key"
        );
        ensure!(!self.rom.as_os_str().is_empty(), "Missing Atari ROM path");
        if let Scoring::Ram { addrs, .. } = &self.scoring {
            ensure!(
                addrs.iter().all(|&addr: &usize| addr < 128),
                "RAM score addresses must be 0..127"
            );
        }
        ensure!(
            !self.min_actions.is_empty()
                && self.min_actions.len() <= ALE_ACTIONS
                && self
                    .min_actions
                    .iter()
                    .all(|&action: &i32| (0..ALE_ACTIONS as i32).contains(&action)),
            "Invalid raw ALE action set"
        );
        ensure!(
            !self.actions.is_empty() && self.actions.len() <= ALE_ACTIONS,
            "Require 1..18 policy actions"
        );
        ensure!(
            self.actions
                .iter()
                .all(|&action: &usize| action < self.min_actions.len()),
            "Action map leaves minimal action set"
        );
        ensure!(
            self.actions
                .iter()
                .enumerate()
                .all(|(idx, action): (usize, &usize)| { !self.actions[..idx].contains(action) }),
            "Duplicate policy actions"
        );
        ensure!(
            (1..=60).contains(&self.frame_skip) && (1..=60).contains(&self.noop_frame_skip),
            "Frame repeat must be 1..60"
        );
        ensure!(
            self.sticky_prob.is_finite()
                && (0.0..=1.0).contains(&self.sticky_prob)
                && self.noop_max <= 100,
            "Invalid reset/sticky settings"
        );
        ensure!(
            self.reset_actions
                .iter()
                .all(|&action: &usize| action < self.min_actions.len()),
            "Invalid reset action"
        );
        ensure!(
            self.max_episode_steps > 0 && (1..=8192).contains(&self.hidden),
            "Invalid network/episode limits"
        );
        Ok(())
    }

    pub fn named(name: &str) -> Result<Self> {
        let normalized: String = name.to_ascii_lowercase().replace(['-', '/', '_', ' '], "");
        let key: &'static str = match normalized.as_str() {
            "pong" | "alepongv5" | "pongv0" | "pongv4" => "pong",
            "breakout" | "alebreakoutv5" => "breakout",
            "boxing" | "aleboxingv5" => "boxing",
            "icehockey" | "aleicehockeyv5" => "icehockey",
            "spaceinvaders" | "alespaceinvadersv5" => "spaceinvaders",
            "tennis" | "aletennisv5" => "tennis",
            _ => bail!("Unknown Atari game {name}; use a full --config file for a custom ROM"),
        };

        let min_actions: Vec<i32> = match key {
            "pong" | "spaceinvaders" => vec![0, 1, 3, 4, 11, 12],
            "breakout" => vec![0, 1, 3, 4],
            _ => (0..ALE_ACTIONS as i32).collect(),
        };
        let actions: Vec<usize> = (0..min_actions.len()).collect();
        let is_sample_factory_compatible: bool = ["boxing", "icehockey", "tennis"].contains(&key);
        let rom_name: &'static str = match key {
            "icehockey" => "ice_hockey",
            "spaceinvaders" => "space_invaders",
            _ => key,
        };
        let scoring: Scoring = match key {
            "boxing" => Scoring::Ram {
                addrs: [18, 19],
                is_bcd: true,
                overrides: vec![[192, 100]],
            },
            "icehockey" => Scoring::Ram {
                addrs: [10, 11],
                is_bcd: true,
                overrides: Vec::new(),
            },
            "tennis" => Scoring::Ram {
                addrs: [71, 72],
                is_bcd: false,
                overrides: Vec::new(),
            },
            _ => Scoring::Signed,
        };
        let reset_actions: Vec<usize> = match key {
            "tennis" => vec![1],
            _ => vec![1, 2],
        };

        Ok(Self {
            key: key.to_owned(),
            rom: PathBuf::from(format!("assets/roms/{rom_name}.bin")),
            scoring,
            actions,
            min_actions,
            grayscale: if is_sample_factory_compatible {
                Grayscale::AfterResize
            } else {
                Grayscale::BeforeResize
            },
            padding: if is_sample_factory_compatible {
                Padding::Repeat
            } else {
                Padding::Zero
            },
            frame_skip: 4,
            sticky_prob: 0.0,
            is_max_pool: true,
            noop_max: 30,
            noop_frame_skip: 1,
            reset_actions,
            is_episodic_life: key == "breakout",
            max_episode_steps: MAX_EPISODE_STEPS,
            hidden: 512,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Alg {
    #[default]
    Dqn,
    Rainbow,
    Iqn,
    Ppo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    #[serde(alias = "algorithm")]
    pub alg: Alg,
    pub iqn_quantiles: usize,
    pub ppo_rollout: usize,
    pub ppo_epochs: usize,
    pub ppo_minibatch: usize,
    pub ppo_clip: f32,
    pub ppo_gae_lambda: f32,
    #[serde(alias = "ppo_value_coef")]
    pub ppo_val_coef: f32,
    pub ppo_entropy_coef: f32,
    pub seed: u64,
    pub steps: u64,
    pub envs: usize,
    #[serde(alias = "replay_capacity")]
    pub replay_cap: usize,
    pub batch_size: usize,
    pub learning_starts: usize,
    pub train_every: u64,
    pub gradient_steps: usize,
    pub target_every: u64,
    #[serde(alias = "learning_rate")]
    pub lr: f64,
    pub gamma: f32,
    pub n_step: usize,
    #[serde(alias = "epsilon_start")]
    pub eps_start: f32,
    #[serde(alias = "epsilon_end")]
    pub eps_end: f32,
    #[serde(alias = "epsilon_decay")]
    pub eps_decay: u64,
    pub priority_alpha: f32,
    pub priority_beta: f32,
    #[serde(alias = "clip_reward")]
    pub is_clip_reward: bool,
    #[serde(alias = "dueling")]
    pub is_dueling: bool,
    #[serde(alias = "evaluate_every")]
    pub eval_every: u64,
    #[serde(alias = "evaluation_episodes")]
    pub eval_episodes: usize,
    #[serde(alias = "evaluation_max_steps")]
    pub eval_max_steps: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            alg: Alg::Dqn,
            iqn_quantiles: 16,
            ppo_rollout: 128,
            ppo_epochs: 4,
            ppo_minibatch: 128,
            ppo_clip: 0.1,
            ppo_gae_lambda: 0.95,
            ppo_val_coef: 0.5,
            ppo_entropy_coef: 0.01,
            seed: 42,
            steps: 50_000_000,
            envs: 4,
            replay_cap: 100_000,
            batch_size: 32,
            learning_starts: 20_000,
            train_every: 4,
            gradient_steps: 1,
            target_every: 2_000,
            lr: 6.25e-5,
            gamma: 0.99,
            n_step: 3,
            eps_start: 1.0,
            eps_end: 0.01,
            eps_decay: 250_000,
            priority_alpha: 0.5,
            priority_beta: 0.4,
            is_clip_reward: true,
            is_dueling: true,
            eval_every: 250_000,
            eval_episodes: 5,
            eval_max_steps: MAX_EPISODE_STEPS,
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=64).contains(&self.iqn_quantiles)
                && (1..=4096).contains(&self.ppo_rollout)
                && (1..=32).contains(&self.ppo_epochs)
                && (1..=8192).contains(&self.ppo_minibatch)
                && (0.0..=1.0).contains(&self.ppo_clip)
                && (0.0..=1.0).contains(&self.ppo_gae_lambda)
                && self.ppo_val_coef.is_finite()
                && self.ppo_val_coef >= 0.0
                && self.ppo_entropy_coef.is_finite()
                && self.ppo_entropy_coef >= 0.0,
            "Invalid IQN/PPO hyperparameters"
        );
        ensure!(
            self.steps > 0 && (1..=256).contains(&self.envs),
            "Invalid steps/env count"
        );
        ensure!(
            self.batch_size > 0
                && self.replay_cap >= self.batch_size
                && self.replay_cap <= 10_000_000,
            "Invalid replay/batch size"
        );
        ensure!(
            self.learning_starts >= self.batch_size && self.learning_starts <= self.replay_cap,
            "learning_starts must be between batch size and replay capacity"
        );
        ensure!(
            self.train_every > 0
                && self.target_every > 0
                && (1..=64).contains(&self.gradient_steps)
                && self.eps_decay > 0,
            "Invalid update/epsilon periods"
        );
        ensure!(
            self.lr.is_finite()
                && self.lr > 0.0
                && self.gamma.is_finite()
                && (0.0..=1.0).contains(&self.gamma),
            "Invalid learning rate/gamma"
        );
        ensure!(
            (1..=32).contains(&self.n_step)
                && self.eps_start.is_finite()
                && self.eps_end.is_finite()
                && (0.0..=1.0).contains(&self.eps_end)
                && self.eps_start >= self.eps_end
                && self.eps_start <= 1.0,
            "Invalid n-step/epsilon"
        );
        ensure!(
            self.priority_alpha.is_finite()
                && self.priority_beta.is_finite()
                && (0.0..=1.0).contains(&self.priority_alpha)
                && (0.0..=1.0).contains(&self.priority_beta),
            "Invalid replay priorities"
        );
        ensure!(
            self.eval_every > 0
                && (1..=256).contains(&self.eval_episodes)
                && self.eval_max_steps > 0,
            "Invalid evaluation settings"
        );
        Ok(())
    }

    pub fn eps(&self, step: u64) -> f32 {
        let warmup: u64 = u64::try_from(self.learning_starts)
            .expect("learning_starts fits u64 on supported targets");
        let decay_step: u64 = step.saturating_sub(warmup).min(self.eps_decay);
        let progress: f64 = decay_step as f64 / self.eps_decay as f64;
        let start: f64 = f64::from(self.eps_start);
        let end: f64 = f64::from(self.eps_end);
        (end + (start - end) * (1.0 - progress)) as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub game: GameSpec,
    #[serde(default)]
    pub training: TrainingConfig,
}

impl RunConfig {
    pub fn named(name: &str) -> Result<Self> {
        Ok(Self {
            game: GameSpec::named(name)?,
            training: TrainingConfig::default(),
        })
    }

    pub fn load_preset(name: &str, dir: &Path) -> Result<Self> {
        let defs: Self = Self::named(name)?;
        let path: PathBuf = dir.join(format!("{}.json", defs.game.key));
        if !path
            .try_exists()
            .with_context(|| format!("Could not inspect preset {}", path.display()))?
        {
            return Ok(defs);
        }
        let cfg: Self = Self::load(&path)?;
        ensure!(
            cfg.game.key == defs.game.key,
            "Preset {} selects {}; expected {}",
            path.display(),
            cfg.game.key,
            defs.game.key
        );
        Ok(cfg)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let input: RunConfigInput = file::read(path)?;
        let game: GameSpec = match input.game {
            GameConfig::Named(name) => GameSpec::named(&name)?,
            GameConfig::Full(spec) => spec,
        };
        let cfg: Self = Self {
            game,
            training: input.training,
        };
        cfg.validate()
            .with_context(|| format!("Invalid configuration values in {}", path.display()))?;
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        file::write(path, self)
    }

    pub fn validate(&self) -> Result<()> {
        self.game.validate()?;
        self.training.validate()
    }
}

pub const GAMES: [&str; 6] = [
    "boxing",
    "breakout",
    "icehockey",
    "pong",
    "spaceinvaders",
    "tennis",
];

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GameConfig {
    Named(String),
    Full(GameSpec),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfigInput {
    game: GameConfig,
    #[serde(default)]
    training: TrainingConfig,
}
