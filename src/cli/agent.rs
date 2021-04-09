use super::{options::ValueFnView, Options, Update, WithUpdate};
use crate::agents::{
    BetaThompsonSamplingAgentConfig, GaePolicyGradientAgentConfig, PolicyGradientAgentConfig,
    TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::defs::AgentDef;
use clap::Clap;

/// Agent name
#[derive(Clap, Debug, Eq, PartialEq, Clone, Copy)]
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

impl<'a, PB, OB> From<&'a Options> for PolicyGradientAgentConfig<PB, OB>
where
    PolicyGradientAgentConfig<PB, OB>: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, PB, OB> Update<&'a Options> for PolicyGradientAgentConfig<PB, OB>
where
    PB: Update<&'a Options>,
    OB: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.policy_config.update(opts);
        self.optimizer_config.update(opts);
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            self.steps_per_epoch = steps_per_epoch;
        }
    }
}

impl<'a, PB, POB, VB, VOB> From<&'a Options> for GaePolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    GaePolicyGradientAgentConfig<PB, POB, VB, VOB>: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, PB, POB, VB, VOB> Update<&'a Options> for GaePolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    PB: Update<&'a Options>,
    POB: Update<&'a Options>,
    VB: Update<&'a Options>,
    VOB: for<'b> Update<&'b ValueFnView<'a>>,
{
    fn update(&mut self, opts: &'a Options) {
        self.policy_config.update(opts);
        self.policy_optimizer_config.update(opts);
        self.value_fn_config.update(opts);
        self.value_fn_optimizer_config.update(&opts.value_fn_view());
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            self.steps_per_epoch = steps_per_epoch;
        }
    }
}
