use super::{Options, Update, WithUpdate};
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
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for TabularQLearningAgentConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(exploration_rate) = opts.exploration_rate {
            self.exploration_rate = exploration_rate;
        }
    }
}

impl From<&Options> for BetaThompsonSamplingAgentConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for BetaThompsonSamplingAgentConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(num_samples) = opts.num_samples {
            self.num_samples = num_samples;
        }
    }
}

impl From<&Options> for UCB1AgentConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for UCB1AgentConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(exploration_rate) = opts.exploration_rate {
            self.exploration_rate = exploration_rate;
        }
    }
}

impl From<&Options> for PolicyGradientAgentDef {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for PolicyGradientAgentDef {
    fn update(&mut self, opts: &Options) {
        self.policy.update(opts);
        self.optimizer.update(opts);
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            self.steps_per_epoch = steps_per_epoch;
        }
    }
}

impl From<&Options> for GaePolicyGradientAgentDef {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for GaePolicyGradientAgentDef {
    fn update(&mut self, opts: &Options) {
        self.policy.update(opts);
        self.policy_optimizer.update(opts);
        self.value_fn.update(opts);
        self.value_fn_optimizer.update(opts);
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            self.steps_per_epoch = steps_per_epoch;
        }
    }
}
