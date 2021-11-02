use super::{CriticDef, CriticUpdaterDef, PolicyDef, PolicyUpdaterDef};
use crate::agents::{
    Agent, BatchUpdateAgentConfig, BetaThompsonSamplingAgentConfig, BuildAgent, BuildAgentError,
    BuildMultithreadAgent, FullAgent, InitializeMultithreadAgent, MultithreadAgentManager,
    MutexAgentConfig, MutexAgentInitializer, RandomAgentConfig, ResettingMetaAgent,
    TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::envs::{EnvStructure, InnerEnvStructure, MetaObservationSpace};
use crate::logging::Loggable;
use crate::spaces::{
    BatchFeatureSpace, ElementRefInto, FeatureSpace, FiniteSpace, ParameterizedDistributionSpace,
    SampleSpace, SendElementSpace, Space,
};
use crate::torch::agents::ActorCriticConfig;
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
    /// Epsilon-greedy tabular Q learning with batch updates.
    BatchTabularQLearning(BatchUpdateAgentConfig<TabularQLearningAgentConfig>),
    /// Thompson sampling of for Bernoulli rewards using Beta priors.
    ///
    /// Assumes no relationship between states.
    BetaThompsonSampling(BetaThompsonSamplingAgentConfig),
    /// UCB1 agent from Auer 2002
    UCB1(UCB1AgentConfig),
    /// Torch actor-critic agent
    ActorCritic(Box<ActorCriticConfig<PolicyDef, PolicyUpdaterDef, CriticDef, CriticUpdaterDef>>),
    /// Applies a non-meta agent to a meta environment by resetting between trials
    ResettingMeta(Box<AgentDef>),
}

/// Multithread agent definition
#[derive(Debug, Clone, PartialEq)]
pub enum MultithreadAgentDef {
    /// A mutex-based simulated multithread agent. Does not provide meaningful parallelism.
    Mutex(Box<AgentDef>),
}

/// A comprehensive space trait for use by RL agents.
///
/// This includes most interfaces required by any agent, environment, or simulator
/// excluding interfaces that can only apply to some spaces, like [`FiniteSpace`].
pub trait RLSpace:
    Space + SendElementSpace + SampleSpace + ElementRefInto<Loggable> + Debug + Send + 'static
{
}
impl<
        T: Space + SendElementSpace + SampleSpace + ElementRefInto<Loggable> + Debug + Send + 'static,
    > RLSpace for T
{
}

/// Comprehensive observation space for use in reinforcement learning
pub trait RLObservationSpace: RLSpace + FeatureSpace<Tensor> + BatchFeatureSpace<Tensor> {}
impl<T: RLSpace + FeatureSpace<Tensor> + BatchFeatureSpace<Tensor>> RLObservationSpace for T {}

/// Comprehensive action space for use in reinforcement learning
pub trait RLActionSpace: RLSpace + ParameterizedDistributionSpace<Tensor> {}
impl<T: RLSpace + ParameterizedDistributionSpace<Tensor>> RLActionSpace for T {}

/// Wrapper implementing [`BuildAgent`] for [`AgentDef`] for any observation and action space.
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

impl<T, OS, AS> BuildAgent<OS, AS> for ForAnyAny<T>
where
    T: Borrow<AgentDef>,
    OS: RLObservationSpace,
    AS: RLActionSpace,
{
    type Agent = Box<DynFullAgent<OS, AS>>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        use AgentDef::*;
        match self.0.borrow() {
            Random => RandomAgentConfig::new()
                .build_agent(env, seed)
                .map(|a| Box::new(a) as _),
            ActorCritic(config) => config
                .as_ref()
                .build_agent(env, seed)
                .map(|a| Box::new(a) as _),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
    }
}

/// [`InitializeMultithreadAgent`] for the agents defined by [`MultithreadAgentDef`].
///
/// This is used instead of `Box<dyn InitializeMultithreadAgent>`
/// because it is not possible to implement `InitializeMultithreadAgent` for the latter
/// because [`InitializeMultithreadAgent::into_manager`] takes the sized `self`.
pub enum GenericMultithreadInitializer<OS: Space, AS: Space> {
    Mutex(MutexAgentInitializer<Box<dyn FullAgent<OS::Element, AS::Element> + Send>>),
}

impl<OS, AS> InitializeMultithreadAgent<OS::Element, AS::Element>
    for GenericMultithreadInitializer<OS, AS>
where
    OS: Space + 'static,
    OS::Element: 'static,
    AS: Space + 'static,
    AS::Element: 'static,
{
    type Manager = Box<dyn MultithreadAgentManager>;
    type Worker = Box<dyn Agent<OS::Element, AS::Element> + Send>;

    fn new_worker(&mut self) -> Self::Worker {
        use GenericMultithreadInitializer::*;
        match self {
            Mutex(initializer) => Box::new(initializer.new_worker()),
        }
    }

    fn into_manager(self) -> Self::Manager {
        use GenericMultithreadInitializer::*;
        match self {
            Mutex(initializer) => Box::new(initializer.into_manager()),
        }
    }
}

impl<T, OS, AS> BuildMultithreadAgent<OS, AS> for ForAnyAny<T>
where
    T: Borrow<MultithreadAgentDef>,
    OS: RLObservationSpace,
    AS: RLActionSpace,
{
    type MultithreadAgent = GenericMultithreadInitializer<OS, AS>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        use MultithreadAgentDef::*;
        Ok(match self.0.borrow() {
            Mutex(config) => GenericMultithreadInitializer::Mutex(
                MutexAgentConfig::new(ForAnyAny(config.borrow()))
                    .build_multithread_agent(env, seed)?,
            ),
        })
    }
}

/// Wrapper implementing [`BuildAgent`] for [`AgentDef`] for finite observation and action spaces.
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

impl<T, OS, AS> BuildAgent<OS, AS> for ForFiniteFinite<T>
where
    T: Borrow<AgentDef>,
    OS: RLObservationSpace + FiniteSpace,
    AS: RLActionSpace + FiniteSpace,
{
    type Agent = Box<DynFullAgent<OS, AS>>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        use AgentDef::*;
        match self.0.borrow() {
            TabularQLearning(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            BatchTabularQLearning(config) => {
                config.build_agent(env, seed).map(|a| Box::new(a) as _)
            }
            BetaThompsonSampling(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            UCB1(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            agent_def => ForAnyAny::new(agent_def).build_agent(env, seed),
        }
    }
}

impl<T, OS, AS> BuildMultithreadAgent<OS, AS> for ForFiniteFinite<T>
where
    T: Borrow<MultithreadAgentDef>,
    OS: RLObservationSpace + FiniteSpace,
    AS: RLActionSpace + FiniteSpace,
{
    type MultithreadAgent = GenericMultithreadInitializer<OS, AS>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        use MultithreadAgentDef::*;
        Ok(match self.0.borrow() {
            Mutex(config) => GenericMultithreadInitializer::Mutex(
                MutexAgentConfig::new(ForFiniteFinite(config.borrow()))
                    .build_multithread_agent(env, seed)?,
            ),
        })
    }
}

/// Wrapper implementing [`BuildAgent`] for [`AgentDef`] for meta finite obs/action spaces.
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

impl<T, OS, AS> BuildAgent<MetaObservationSpace<OS, AS>, AS> for ForMetaFiniteFinite<T>
where
    T: Borrow<AgentDef>,
    OS: RLObservationSpace + FiniteSpace + Clone,
    OS::Element: Clone,
    AS: RLActionSpace + FiniteSpace + Clone,
    AS::Element: Clone,
    // I think this ought to be inferrable but for some reason it isn't
    // It may have to do with SendElementSpace
    MetaObservationSpace<OS, AS>: RLObservationSpace,
{
    type Agent = Box<DynFullAgent<MetaObservationSpace<OS, AS>, AS>>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        use AgentDef::*;

        match self.0.borrow() {
            ResettingMeta(inner_agent_def) => ResettingMetaAgent::new(
                ForFiniteFinite::new(inner_agent_def.as_ref().clone()),
                (&InnerEnvStructure::new(env)).into(),
                seed,
            )
            .map(|a| Box::new(a) as _),
            agent_def => ForAnyAny::new(agent_def).build_agent(env, seed),
        }
    }
}

impl<T, OS, AS> BuildMultithreadAgent<MetaObservationSpace<OS, AS>, AS> for ForMetaFiniteFinite<T>
where
    T: Borrow<MultithreadAgentDef>,
    OS: RLObservationSpace + FiniteSpace + Clone,
    OS::Element: Clone,
    AS: RLActionSpace + FiniteSpace + Clone,
    AS::Element: Clone,
    // I think this ought to be inferrable but for some reason it isn't
    // It may have to do with SendElementSpace
    MetaObservationSpace<OS, AS>: RLObservationSpace,
{
    type MultithreadAgent = GenericMultithreadInitializer<MetaObservationSpace<OS, AS>, AS>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        use MultithreadAgentDef::*;
        Ok(match self.0.borrow() {
            Mutex(config) => GenericMultithreadInitializer::Mutex(
                MutexAgentConfig::new(ForMetaFiniteFinite(config.borrow()))
                    .build_multithread_agent(env, seed)?,
            ),
        })
    }
}

/// Send-able [`FullAgent`] trait object for an environment structure.
pub type DynFullAgent<OS, AS> =
    dyn FullAgent<<OS as Space>::Element, <AS as Space>::Element> + Send;
