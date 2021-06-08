use super::{options::ValueFnView, Options, Update, WithUpdate};
use crate::agents::{
    BetaThompsonSamplingAgentConfig, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::defs::AgentDef;
use crate::torch::agents::{PolicyGradientAgentConfig, PolicyValueNetActorConfig, TrpoAgentConfig};
use clap::{ArgEnum, Clap};
use std::fmt;
use std::str::FromStr;

/// Concrete agent type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Clap)]
pub enum ConcreteAgentType {
    Random,
    TabularQLearning,
    BetaThompsonSampling,
    UCB1,
    PolicyGradient,
    Trpo,
}

impl fmt::Display for ConcreteAgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Self::VARIANTS[*self as usize])
    }
}

/// Wrapper agent type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Clap)]
pub enum AgentWrapperType {
    ResettingMeta,
}

impl fmt::Display for AgentWrapperType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Self::VARIANTS[*self as usize])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AgentType {
    /// Base concrete agent
    pub base: ConcreteAgentType,
    /// Agent wrappers; applied right to left
    pub wrappers: Vec<AgentWrapperType>,
}

impl fmt::Display for AgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for wrapper in &self.wrappers {
            write!(f, "{}:", wrapper)?;
        }
        write!(f, "{}", self.base)
    }
}

impl FromStr for AgentType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let case_insensitive = true;
        if let Some((wrapper_str, base_str)) = s.rsplit_once(':') {
            let base = ConcreteAgentType::from_str(base_str, case_insensitive)?;
            let wrappers = wrapper_str
                .split(':')
                .map(|ws| AgentWrapperType::from_str(ws, case_insensitive))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self { base, wrappers })
        } else {
            Ok(Self {
                base: ConcreteAgentType::from_str(s, case_insensitive)?,
                wrappers: Vec::new(),
            })
        }
    }
}

impl From<&Options> for AgentDef {
    fn from(opts: &Options) -> Self {
        use AgentWrapperType::*;
        use ConcreteAgentType::*;
        let mut agent_def = match opts.agent.base {
            Random => Self::Random,
            TabularQLearning => Self::TabularQLearning(opts.into()),
            BetaThompsonSampling => Self::BetaThompsonSampling(opts.into()),
            UCB1 => Self::UCB1(From::from(opts)),
            PolicyGradient => Self::PolicyGradient(Box::new(opts.into())),
            Trpo => Self::Trpo(Box::new(opts.into())),
        };
        for wrapper in opts.agent.wrappers.iter().rev() {
            agent_def = match wrapper {
                ResettingMeta => Self::ResettingMeta(Box::new(agent_def)),
            };
        }
        agent_def
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

impl<'a, PB, POB, VB, VOB> From<&'a Options> for TrpoAgentConfig<PB, POB, VB, VOB>
where
    Self: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, PB, POB, VB, VOB> Update<&'a Options> for TrpoAgentConfig<PB, POB, VB, VOB>
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
        if let Some(max_policy_step_kl) = opts.max_policy_step_kl {
            self.max_policy_step_kl = max_policy_step_kl;
        }
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
        if let Some(device) = opts.device {
            self.device = device.into();
        }
    }
}
