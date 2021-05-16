use super::{options::ValueFnView, Options, Update, WithUpdate};
use crate::agents::{
    BetaThompsonSamplingAgentConfig, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::defs::AgentDef;
use crate::torch::agents::{PolicyGradientAgentConfig, PolicyValueNetActorConfig};
use clap::Clap;

/// Agent name
#[derive(Clap, Debug, Eq, PartialEq, Clone, Copy)]
pub enum AgentType {
    Random,
    TabularQLearning,
    BetaThompsonSampling,
    UCB1,
    PolicyGradient,
}

impl From<&Options> for AgentDef {
    fn from(opts: &Options) -> Self {
        use AgentType::*;
        match opts.agent {
            Random => Self::Random,
            TabularQLearning => Self::TabularQLearning(opts.into()),
            BetaThompsonSampling => Self::BetaThompsonSampling(opts.into()),
            UCB1 => Self::UCB1(From::from(opts)),
            PolicyGradient => Self::PolicyGradient(Box::new(opts.into())),
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

impl<'a, PB, POB, VB, VOB> From<&'a Options> for PolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    Self: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, PB, POB, VB, VOB> Update<&'a Options> for PolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    PB: Update<&'a Options>,
    POB: Update<&'a Options>,
    VB: Update<&'a Options>,
    VOB: for<'b> Update<&'b ValueFnView<'a>>,
{
    fn update(&mut self, opts: &'a Options) {
        self.actor_config.update(opts);
        self.policy_optimizer_config.update(opts);
        self.value_optimizer_config.update(&opts.value_fn_view());
    }
}

impl<'a, PB, VB> From<&'a Options> for PolicyValueNetActorConfig<PB, VB>
where
    Self: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, PB, VB> Update<&'a Options> for PolicyValueNetActorConfig<PB, VB>
where
    PB: Update<&'a Options>,
    VB: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.policy_config.update(opts);
        self.value_config.update(opts);
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            self.steps_per_epoch = steps_per_epoch;
        }
        if let Some(value_train_iters) = opts.value_fn_train_iters {
            self.value_train_iters = value_train_iters;
        }
    }
}
