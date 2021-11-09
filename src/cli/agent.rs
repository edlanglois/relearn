use super::{Options, Update, WithUpdate};
use crate::agents::{
    buffers::SerialBufferConfig, BatchUpdateAgentConfig, BetaThompsonSamplingAgentConfig,
    MultithreadBatchAgentConfig, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::defs::{
    AgentDef, BatchActorDef, CriticDef, CriticUpdaterDef, MultithreadAgentDef,
    OptionalBatchAgentDef, PolicyDef, PolicyUpdaterDef,
};
use crate::torch::agents::ActorCriticConfig;
use clap::ArgEnum;
use std::fmt;
use std::str::FromStr;

/// Concrete agent type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum ConcreteAgentType {
    Random,
    TabularQLearning,
    BetaThompsonSampling,
    UCB1,
    PolicyGradient,
    Trpo,
    Ppo,
}

impl fmt::Display for ConcreteAgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_arg_value().unwrap().get_name())
    }
}

impl ConcreteAgentType {
    pub fn optional_batch_agent_def(&self, opts: &Options) -> Option<OptionalBatchAgentDef> {
        use ConcreteAgentType::*;
        match self {
            Random => Some(OptionalBatchAgentDef::Random),
            TabularQLearning => Some(OptionalBatchAgentDef::TabularQLearning(opts.into())),
            BetaThompsonSampling => Some(OptionalBatchAgentDef::BetaThompsonSampling(opts.into())),
            UCB1 => Some(OptionalBatchAgentDef::UCB1(From::from(opts))),
            _ => None,
        }
    }

    pub fn agent_def(&self, opts: &Options) -> AgentDef {
        use ConcreteAgentType::*;
        match self {
            PolicyGradient => {
                let config = ActorCriticConfig {
                    policy_updater_config: PolicyUpdaterDef::default_policy_gradient(),
                    ..ActorCriticConfig::default()
                }
                .with_update(opts);
                AgentDef::ActorCritic(Box::new(config))
            }
            Trpo => {
                let config = ActorCriticConfig {
                    policy_updater_config: PolicyUpdaterDef::default_trpo(),
                    ..ActorCriticConfig::default()
                }
                .with_update(opts);
                AgentDef::ActorCritic(Box::new(config))
            }
            Ppo => {
                let config = ActorCriticConfig {
                    policy_updater_config: PolicyUpdaterDef::default_ppo(),
                    ..ActorCriticConfig::default()
                }
                .with_update(opts);
                AgentDef::ActorCritic(Box::new(config))
            }
            _ => AgentDef::NoBatch(self.optional_batch_agent_def(opts).unwrap()),
        }
    }
}

/// Wrapper agent type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum AgentWrapperType {
    ResettingMeta,
    Mutex,
    Batch,
}

impl fmt::Display for AgentWrapperType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_arg_value().unwrap().get_name())
    }
}

impl AgentWrapperType {
    pub fn agent_def(&self, inner: AgentDef, opts: &Options) -> Option<AgentDef> {
        use AgentWrapperType::*;
        match self {
            ResettingMeta => Some(AgentDef::ResettingMeta(Box::new(inner))),
            Batch => {
                if let AgentDef::NoBatch(optional_batch_agent_def) = inner {
                    Some(AgentDef::Batch(BatchUpdateAgentConfig {
                        actor_config: BatchActorDef::Batch(optional_batch_agent_def),
                        history_buffer_config: opts.into(),
                    }))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn multi_thread_agent_def(
        &self,
        inner: AgentDef,
        opts: &Options,
    ) -> Option<MultithreadAgentDef> {
        use AgentWrapperType::*;
        match self {
            Mutex => Some(MultithreadAgentDef::Mutex(inner)),
            Batch => {
                if let AgentDef::NoBatch(optional_batch_agent_def) = inner {
                    Some(MultithreadAgentDef::Batch(MultithreadBatchAgentConfig {
                        actor_config: BatchActorDef::Batch(optional_batch_agent_def),
                        buffer_config: opts.into(),
                    }))
                } else {
                    None
                }
            }
            _ => None,
        }
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

impl AgentType {
    pub fn agent_def(&self, opts: &Options) -> Option<AgentDef> {
        let mut agent_def = self.base.agent_def(opts);
        for wrapper in self.wrappers.iter().rev() {
            agent_def = wrapper.agent_def(agent_def, opts)?;
        }
        Some(agent_def)
    }

    pub fn multi_thread_agent_def(&self, opts: &Options) -> Option<MultithreadAgentDef> {
        if let Some(outer_wrapper) = self.wrappers.first() {
            let inner = Self {
                base: self.base,
                wrappers: self.wrappers[1..].into(),
            };
            let inner_agent = inner.agent_def(opts)?;
            outer_wrapper.multi_thread_agent_def(inner_agent, opts)
        } else {
            None
        }
    }
}

impl From<&Options> for Option<AgentDef> {
    fn from(opts: &Options) -> Self {
        opts.agent.agent_def(opts)
    }
}

impl From<&Options> for Option<MultithreadAgentDef> {
    fn from(opts: &Options) -> Self {
        opts.agent.multi_thread_agent_def(opts)
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

impl<'a, AC> From<&'a Options> for BatchUpdateAgentConfig<AC>
where
    AC: From<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self {
            actor_config: opts.into(),
            history_buffer_config: opts.into(),
        }
    }
}

impl<'a, AC> Update<&'a Options> for BatchUpdateAgentConfig<AC>
where
    AC: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.actor_config.update(opts);
        self.history_buffer_config.update(opts);
    }
}

impl From<&Options> for SerialBufferConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for SerialBufferConfig {
    fn update(&mut self, _opts: &Options) {
        // TODO: Allow setting thresholds
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

impl Update<&Options>
    for ActorCriticConfig<PolicyDef, PolicyUpdaterDef, CriticDef, CriticUpdaterDef>
{
    fn update(&mut self, opts: &Options) {
        self.policy_config.update(opts);
        self.policy_updater_config.update(opts);
        self.critic_config.update(opts);
        self.critic_updater_config.update(opts);
        if let Some(steps_per_epoch) = opts.steps_per_epoch {
            self.steps_per_epoch = steps_per_epoch;
        }
        if let Some(device) = opts.device {
            self.device = device.into();
        }
    }
}
