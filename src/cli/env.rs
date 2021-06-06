//! Parse environment definition from Options
use super::{Options, Update, WithUpdate};
use crate::defs::{env::DistributionType, EnvDef};
use crate::envs::{
    Chain, FixedMeansBanditConfig, MemoryGame, MetaEnvConfig, OneHotBandits,
    PriorMeansBanditConfig, UniformBernoulliBandits,
};
use clap::Clap;
use rand::distributions::Standard;
use std::convert::TryInto;

/// Environment name
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Clap)]
pub enum EnvType {
    DeterministicBandit,
    BernoulliBandit,
    Chain,
    MemoryGame,
    MetaOneHotBandits,
    MetaUniformBernoulliBandits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Clap)]
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
