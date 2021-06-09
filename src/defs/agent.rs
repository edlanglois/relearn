use super::{OptimizerDef, SeqModDef, StepValueDef};
use crate::agents::{
    Agent, AgentBuilder, BetaThompsonSamplingAgentConfig, BuildAgentError, RandomAgentConfig,
    ResettingMetaAgent, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::envs::{EnvStructure, InnerEnvStructure, MetaObservationSpace};
use crate::logging::Loggable;
use crate::spaces::{
    BatchFeatureSpace, ElementRefInto, FeatureSpace, FiniteSpace, ParameterizedDistributionSpace,
    SampleSpace, Space,
};
use crate::torch::agents::{
    PolicyGradientAgentConfig, PolicyGradientBoxedAgent, TrpoAgentConfig, TrpoBoxedAgent,
};
use crate::torch::optimizers::ConjugateGradientOptimizerConfig;
use std::borrow::Borrow;
use std::fmt::Debug;
use tch::Tensor;

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
    /// Applies a non-meta agent to a meta environment by resetting between trials
    ResettingMeta(Box<AgentDef>),
}

/// A comprehensive space trait for use by RL agents.
///
/// This includes most interfaces required by any agent, environment, or simulator
/// excluding interfaces that can only apply to some spaces, like [`FiniteSpace`].
pub trait RLSpace: Space + SampleSpace + ElementRefInto<Loggable> + Debug {}
impl<T: Space + SampleSpace + ElementRefInto<Loggable> + Debug> RLSpace for T {}

/// Comprehensive observation space for use in reinforcement learning
pub trait RLObservationSpace: RLSpace + FeatureSpace<Tensor> + BatchFeatureSpace<Tensor> {}
impl<T: RLSpace + FeatureSpace<Tensor> + BatchFeatureSpace<Tensor>> RLObservationSpace for T {}

/// Comprehensive action space for use in reinforcement learning
pub trait RLActionSpace: RLSpace + ParameterizedDistributionSpace<Tensor> {}
impl<T: RLSpace + ParameterizedDistributionSpace<Tensor>> RLActionSpace for T {}

/// Wrapper implementing [`AgentBuilder`] for [`AgentDef`] for any observation and action space.
///
/// More specifically, any observation and action space satisfying the relatively generic
/// [`RLObservationSpace`] and [`RLActionSpace`] traits.
///
/// There is no trait specialization so this will fail for those agents that require a tighter
/// bounds on the observation and actions paces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForAnyAny<T>(T);

impl<T> ForAnyAny<T> {
    pub const fn new(agent_def: T) -> Self {
        Self(agent_def)
    }
}

impl<T, E> AgentBuilder<Box<DynEnvAgent<E>>, E> for ForAnyAny<T>
where
    T: Borrow<AgentDef>,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: RLObservationSpace + 'static,
    <E as EnvStructure>::ActionSpace: RLActionSpace + 'static,
{
    fn build_agent(&self, env: &E, seed: u64) -> Result<Box<DynEnvAgent<E>>, BuildAgentError> {
        use AgentDef::*;
        match self.0.borrow() {
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

/// Wrapper implementing [`AgentBuilder`] for [`AgentDef`] for finite observation and action spaces.
///
/// There is no trait specialization so this will fail for those agents that require a tighter
/// bounds on the observation and actions paces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForFiniteFinite<T>(T);

impl<T> ForFiniteFinite<T> {
    pub const fn new(agent_def: T) -> Self {
        Self(agent_def)
    }
}

impl<T, E> AgentBuilder<Box<DynEnvAgent<E>>, E> for ForFiniteFinite<T>
where
    T: Borrow<AgentDef>,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: RLObservationSpace + FiniteSpace + 'static,
    <E as EnvStructure>::ActionSpace: RLActionSpace + FiniteSpace + 'static,
{
    fn build_agent(&self, env: &E, seed: u64) -> Result<Box<DynEnvAgent<E>>, BuildAgentError> {
        use AgentDef::*;
        match self.0.borrow() {
            TabularQLearning(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            BetaThompsonSampling(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            UCB1(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            agent_def => ForAnyAny::new(agent_def).build_agent(env, seed),
        }
    }
}

/// Wrapper implementing [`AgentBuilder`] for [`AgentDef`] for meta finite obs/action spaces.
///
/// Specifically, it is the inner observation space that must be finite.
/// The outer observation space is not finite.
///
/// There is no trait specialization so this will fail for those agents that require a tighter
/// bounds on the observation and actions paces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForMetaFiniteFinite<T>(T);

impl<T> ForMetaFiniteFinite<T> {
    pub const fn new(agent_def: T) -> Self {
        Self(agent_def)
    }
}

impl<T, E, OS, AS> AgentBuilder<Box<DynEnvAgent<E>>, E> for ForMetaFiniteFinite<T>
where
    T: Borrow<AgentDef>,
    E: EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS> + ?Sized,
    <E as EnvStructure>::ObservationSpace: RLObservationSpace + 'static,
    OS: RLObservationSpace + FiniteSpace + Clone + 'static,
    <OS as Space>::Element: Clone,
    AS: RLActionSpace + FiniteSpace + Clone + 'static,
    <AS as Space>::Element: Clone,
{
    fn build_agent(&self, env: &E, seed: u64) -> Result<Box<DynEnvAgent<E>>, BuildAgentError> {
        use AgentDef::*;

        match self.0.borrow() {
            ResettingMeta(inner_agent_def) => Ok(Box::new(ResettingMetaAgent::new(
                ForFiniteFinite::new(inner_agent_def.as_ref().clone()),
                (&InnerEnvStructure::<E, &E>::new(env)).into(),
                seed,
            )) as _),
            agent_def => ForAnyAny::new(agent_def).build_agent(env, seed),
        }
    }
}

// TODO: Return Box<dyn ActorAgent> where ActorAgent: Actor + Agent instead of Box<dyn Agent>

/// The agent trait object for a given environment structure.
pub type DynEnvAgent<E> = dyn Agent<
    <<E as EnvStructure>::ObservationSpace as Space>::Element,
    <<E as EnvStructure>::ActionSpace as Space>::Element,
>;
