use super::Options;
use crate::agents::{
    BetaThompsonSamplingAgentConfig, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::defs::{AgentDef, GaePolicyGradientAgentDef, PolicyGradientAgentDef};
use clap::Clap;

/// Agent name
#[derive(Clap, Debug)]
pub enum AgentType {
    Random,
    TabularQLearning,
    BetaThompsonSampling,
    UCB1,
    PolicyGradient,
    GaePolicyGradient,
}

impl From<&Options> for AgentDef {
    fn from(opts: &Options) -> Self {
        use AgentType::*;
        match opts.agent {
            Random => AgentDef::Random,
            TabularQLearning => AgentDef::TabularQLearning(opts.into()),
            BetaThompsonSampling => AgentDef::BetaThompsonSampling(opts.into()),
            UCB1 => AgentDef::UCB1(From::from(opts)),
            PolicyGradient => AgentDef::PolicyGradient(opts.into()),
            GaePolicyGradient => AgentDef::GaePolicyGradient(opts.into()),
        }
    }
}

impl From<&Options> for TabularQLearningAgentConfig {
    fn from(opts: &Options) -> Self {
        let mut config = TabularQLearningAgentConfig::default();
        if let Some(exploration_rate) = opts.exploration_rate {
            config.exploration_rate = exploration_rate;
        }
        config
    }
}

impl From<&Options> for BetaThompsonSamplingAgentConfig {
    fn from(opts: &Options) -> Self {
        let mut config = BetaThompsonSamplingAgentConfig::default();
        if let Some(num_samples) = opts.num_samples {
            config.num_samples = num_samples;
        }
        config
    }
}

impl From<&Options> for UCB1AgentConfig {
    fn from(opts: &Options) -> Self {
        let mut config = UCB1AgentConfig::default();
        if let Some(exploration_rate) = opts.exploration_rate {
            config.exploration_rate = exploration_rate;
        }
        config
    }
}

impl From<&Options> for PolicyGradientAgentDef {
    fn from(opts: &Options) -> Self {
        let mut config = PolicyGradientAgentDef::default();
        config.policy = opts.into();
        config.optimizer = opts.into();
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            config.steps_per_epoch = steps_per_epoch;
        }
        config
    }
}

impl From<&Options> for GaePolicyGradientAgentDef {
    fn from(opts: &Options) -> Self {
        let mut config = GaePolicyGradientAgentDef::default();
        config.policy = opts.into();
        config.policy_optimizer = opts.into();
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            config.steps_per_epoch = steps_per_epoch;
        }
        config
    }
}
