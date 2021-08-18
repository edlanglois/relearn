//! Parse environment definition from Options
use super::{Options, Update, WithUpdate};
use crate::defs::{env::DistributionType, EnvDef};
use crate::envs::{
    Chain, DirichletRandomMdps, FixedMeansBanditConfig, MemoryGame, MetaEnvConfig, OneHotBandits,
    PriorMeansBanditConfig, StepLimit, UniformBernoulliBandits, Wrapped,
};
use clap::ArgEnum;
use rand::distributions::Standard;
use std::convert::TryInto;

/// Environment name
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum EnvType {
    DeterministicBandit,
    BernoulliBandit,
    Chain,
    MemoryGame,
    MetaOneHotBandits,
    MetaUniformBernoulliBandits,
    MetaDirichletMdps,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum BanditArmPrior {
    Fixed,
    Uniform,
}

impl From<&Options> for EnvDef {
    fn from(opts: &Options) -> Self {
        use EnvType::*;
        match opts.environment {
            DeterministicBandit => bandit_env_def(DistributionType::Deterministic, opts),
            BernoulliBandit => bandit_env_def(DistributionType::Bernoulli, opts),
            Chain => Self::Chain(opts.into()),
            MemoryGame => Self::MemoryGame(opts.into()),
            MetaOneHotBandits => Self::MetaOneHotBandits(opts.into()),
            MetaUniformBernoulliBandits => Self::MetaUniformBernoulliBandits(opts.into()),
            MetaDirichletMdps => Self::MetaDirichletMdps(opts.into()),
        }
    }
}

fn bandit_env_def(sample_distribution: DistributionType, opts: &Options) -> EnvDef {
    match opts.arm_prior {
        BanditArmPrior::Fixed => EnvDef::FixedMeanBandit(sample_distribution, opts.into()),
        BanditArmPrior::Uniform => EnvDef::UniformMeanBandit(sample_distribution, opts.into()),
    }
}

impl From<&Options> for FixedMeansBanditConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for FixedMeansBanditConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(ref arm_rewards) = opts.arm_rewards {
            self.means = arm_rewards.clone();
        }
    }
}

impl From<&Options> for PriorMeansBanditConfig<Standard> {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for PriorMeansBanditConfig<Standard> {
    fn update(&mut self, opts: &Options) {
        if let Some(num_actions) = opts.num_actions {
            self.num_arms = num_actions.try_into().unwrap();
        }
    }
}

impl From<&Options> for Chain {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for Chain {
    fn update(&mut self, opts: &Options) {
        if let Some(num_states) = opts.num_states {
            self.size = num_states;
        }
        if let Some(discount_factor) = opts.discount_factor {
            self.discount_factor = discount_factor;
        }
    }
}

impl From<&Options> for MemoryGame {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for MemoryGame {
    fn update(&mut self, opts: &Options) {
        if let Some(num_actions) = opts.num_actions {
            self.num_actions = num_actions.try_into().unwrap();
        }
        if let Some(episode_len) = opts.episode_len {
            self.history_len = (episode_len - 1).try_into().unwrap();
        }
    }
}

impl<'a, EB> From<&'a Options> for MetaEnvConfig<EB>
where
    Self: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, EB> Update<&'a Options> for MetaEnvConfig<EB>
where
    EB: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.env_dist_config.update(opts);
        if let Some(num_episodes_per_trial) = opts.episodes_per_trial {
            self.num_episodes_per_trial = num_episodes_per_trial;
        }
    }
}

impl<'a, T, W> From<&'a Options> for Wrapped<T, W>
where
    T: From<&'a Options>,
    W: From<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self {
            inner: opts.into(),
            wrapper: opts.into(),
        }
    }
}

impl<'a, T, W> Update<&'a Options> for Wrapped<T, W>
where
    T: Update<&'a Options>,
    W: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.inner.update(opts);
        self.wrapper.update(opts);
    }
}

impl From<&Options> for StepLimit {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for StepLimit {
    fn update(&mut self, opts: &Options) {
        if let Some(max_steps_per_episode) = opts.max_steps_per_episode {
            self.max_steps_per_episode = max_steps_per_episode;
        }
    }
}

impl From<&Options> for UniformBernoulliBandits {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for UniformBernoulliBandits {
    fn update(&mut self, opts: &Options) {
        if let Some(num_actions) = opts.num_actions {
            self.num_arms = num_actions.try_into().unwrap();
        }
    }
}

impl From<&Options> for OneHotBandits {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for OneHotBandits {
    fn update(&mut self, opts: &Options) {
        if let Some(num_actions) = opts.num_actions {
            self.num_arms = num_actions.try_into().unwrap();
        }
    }
}

impl From<&Options> for DirichletRandomMdps {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for DirichletRandomMdps {
    fn update(&mut self, opts: &Options) {
        if let Some(num_states) = opts.num_states {
            self.num_states = num_states.try_into().unwrap();
        }
        if let Some(num_actions) = opts.num_actions {
            self.num_actions = num_actions.try_into().unwrap();
        }
        if let Some(discount_factor) = opts.discount_factor {
            self.discount_factor = discount_factor;
        }
    }
}
