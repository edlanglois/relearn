use super::{OptimizerDef, SeqModDef, StepValueDef};
use crate::agents::{
    Agent, AgentBuilder, BetaThompsonSamplingAgentConfig, BuildAgentError, RandomAgentConfig,
    TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::envs::EnvStructure;
use crate::spaces::{FiniteSpace, RLActionSpace, RLObservationSpace, Space};
use crate::torch::agents::{
    PolicyGradientAgentConfig, PolicyGradientBoxedAgent, TrpoAgentConfig, TrpoBoxedAgent,
};
use crate::torch::optimizers::ConjugateGradientOptimizerConfig;
use std::fmt::Debug;

/// Agent definition
#[derive(Debug, Clone, PartialEq)]
pub enum AgentDef {
    /// An agent that selects actions randomly.
    Random,
    /// Epsilon-greedy tabular Q learning.
    TabularQLearning(TabularQLearningAgentConfig),
    /// Thompson sampling of for Bernoulli rewards using Beta priors.
    ///
    /// Assumes no relationship between states.
    BetaThompsonSampling(BetaThompsonSamplingAgentConfig),
    /// UCB1 agent from Auer 2002
    UCB1(UCB1AgentConfig),
    /// Policy gradient
    PolicyGradient(
        Box<PolicyGradientAgentConfig<SeqModDef, OptimizerDef, StepValueDef, OptimizerDef>>,
    ),
    /// Trust region policy optimizer
    Trpo(
        Box<
            TrpoAgentConfig<
                SeqModDef,
                ConjugateGradientOptimizerConfig,
                StepValueDef,
                OptimizerDef,
            >,
        >,
    ),
}

// TODO: Return Box<dyn ActorAgent> where ActorAgent: Actor + Agent instead of Box<dyn Agent>

/// The agent trait object for a given environment structure.
pub type DynEnvAgent<E> = dyn Agent<
    <<E as EnvStructure>::ObservationSpace as Space>::Element,
    <<E as EnvStructure>::ActionSpace as Space>::Element,
>;

impl AgentDef {
    /// Construct an agent for an environment with finite observation and action spaces.
    pub fn build_finite_finite<E>(
        &self,
        env: &E,
        seed: u64,
    ) -> Result<Box<DynEnvAgent<E>>, BuildAgentError>
    where
        E: EnvStructure + ?Sized,
        <E as EnvStructure>::ObservationSpace: RLObservationSpace + FiniteSpace + 'static,
        <E as EnvStructure>::ActionSpace: RLActionSpace + FiniteSpace + 'static,
    {
        use AgentDef::*;
        match self {
            TabularQLearning(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            BetaThompsonSampling(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            UCB1(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            _ => self.build_any_any(env, seed),
        }
    }

    /// Construct an agent for an environment with arbitrary observation and action spaces.
    ///
    /// There is no trait specialization so this will fail if the agent cannot be built for
    /// arbitrary spaces, even if it can for the specific instance this function is called with.
    pub fn build_any_any<E>(
        &self,
        env: &E,
        seed: u64,
    ) -> Result<Box<DynEnvAgent<E>>, BuildAgentError>
    where
        E: EnvStructure + ?Sized,
        <E as EnvStructure>::ObservationSpace: RLObservationSpace + 'static,
        <E as EnvStructure>::ActionSpace: RLActionSpace + 'static,
    {
        use AgentDef::*;
        match self {
            Random => RandomAgentConfig::new()
                .build_agent(env, seed)
                .map(|a| Box::new(a) as _),
            PolicyGradient(config) => config
                .as_ref()
                .build_agent(env, seed)
                .map(|a: PolicyGradientBoxedAgent<_, _>| Box::new(a) as _),
            Trpo(config) => config
                .as_ref()
                .build_agent(env, seed)
                .map(|a: TrpoBoxedAgent<_, _>| Box::new(a) as _),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
    }
}
