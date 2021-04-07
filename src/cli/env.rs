//! Parse environment definition from Options
use super::Options;
use crate::defs::{env::DistributionType, EnvDef};
use crate::envs::{Chain, FixedMeansBanditConfig, PriorMeansBanditConfig};
use clap::Clap;
use rand::distributions::Standard;

/// Environment name
#[derive(Clap, Debug)]
pub enum EnvName {
    DeterministicBandit,
    BernoulliBandit,
    Chain,
}

#[derive(Clap, Debug)]
pub enum BanditArmPrior {
    Fixed,
    Uniform,
}

impl From<&Options> for EnvDef {
    fn from(opts: &Options) -> Self {
        use EnvName::*;
        match opts.environment {
            DeterministicBandit => bandit_env_def(DistributionType::Deterministic, opts),
            BernoulliBandit => bandit_env_def(DistributionType::Bernoulli, opts),
            Chain => EnvDef::Chain(opts.into()),
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
        let mut config = Self::default();
        if let Some(ref arm_rewards) = opts.arm_rewards {
            config.means = arm_rewards.clone();
        }
        config
    }
}

impl From<&Options> for PriorMeansBanditConfig<Standard> {
    fn from(opts: &Options) -> Self {
        let mut config = Self::default();
        if let Some(num_actions) = opts.num_actions {
            config.num_arms = num_actions as usize;
        }
        config
    }
}

impl From<&Options> for Chain {
    fn from(opts: &Options) -> Self {
        let mut config = Self::default();
        if let Some(num_states) = opts.num_states {
            config.size = num_states;
        }
        if let Some(discount_factor) = opts.discount_factor {
            config.discount_factor = discount_factor;
        }
        config
    }
}
