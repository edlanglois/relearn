use super::{CriticDef, CriticUpdaterDef, PolicyDef, PolicyUpdaterDef};
use crate::agents::{
    multithread::MutexAgentInitializer, Agent, BatchUpdateAgentConfig,
    BetaThompsonSamplingAgentConfig, BuildAgent, BuildAgentError, BuildMultithreadAgent, FullAgent,
    InitializeMultithreadAgent, MultithreadAgentManager, MutexAgentConfig, RandomAgentConfig,
    ResettingMetaAgent, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::envs::{EnvStructure, InnerEnvStructure, MetaObservationSpace};
use crate::logging::Loggable;
use crate::spaces::{
    BatchFeatureSpace, ElementRefInto, FeatureSpace, FiniteSpace, ParameterizedDistributionSpace,
    SampleSpace, SendElementSpace, Space,
};
use crate::torch::agents::ActorCriticConfig;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
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

/// Environment structure marker types
pub trait EnvStructureMarker {}

/// Marker for environments with any observation and action space.
///
/// More specifically, any observation and action space satisfying the relatively generic
/// [`RLObservationSpace`] and [`RLActionSpace`] traits.
///
/// There is no trait specialization so construction will fail for those agents that require
/// a tighter bound on the observation and action spaces.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)] // Simplify deriving trait for structs using it
pub enum EnvAnyAny {}
impl EnvStructureMarker for EnvAnyAny {}

/// Marker for environments with finite observation and action spaces.
///
/// There is no trait specialization so construction will fail for those agents that require
/// a tighter bound on the observation and action spaces.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)] // Simplify deriving trait for structs using it
pub enum EnvFiniteFinite {}
impl EnvStructureMarker for EnvFiniteFinite {}

/// Marker for meta-environments with finite inner observation and action spaces.
///
/// There is no trait specialization so construction will fail for those agents that require
/// a tighter bound on the observation and action spaces.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)] // Simplify deriving trait for structs using it
pub enum EnvMetaFiniteFinite {}
impl EnvStructureMarker for EnvMetaFiniteFinite {}

/// Build an agent for environments satisfying the marker type.
pub trait BuildAgentFor<M, OS: Space, AS: Space> {
    type Agent: FullAgent<OS::Element, AS::Element>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError>;
}

impl<T, M, OS, AS> BuildAgentFor<M, OS, AS> for T
where
    T: Deref + ?Sized,
    T::Target: BuildAgentFor<M, OS, AS>,
    OS: Space,
    AS: Space,
{
    type Agent = <T::Target as BuildAgentFor<M, OS, AS>>::Agent;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        self.deref().build_agent(env, seed)
    }
}

impl<OS, AS> BuildAgentFor<EnvAnyAny, OS, AS> for AgentDef
where
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
        match self {
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

impl<OS, AS> BuildAgentFor<EnvFiniteFinite, OS, AS> for AgentDef
where
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
        match self {
            TabularQLearning(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            BatchTabularQLearning(config) => {
                config.build_agent(env, seed).map(|a| Box::new(a) as _)
            }
            BetaThompsonSampling(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            UCB1(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            _ => BuildAgentFor::<EnvAnyAny, _, _>::build_agent(self, env, seed),
        }
    }
}

impl<OS, AS> BuildAgentFor<EnvMetaFiniteFinite, MetaObservationSpace<OS, AS>, AS> for AgentDef
where
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

        match self {
            ResettingMeta(inner_agent_def) => ResettingMetaAgent::new(
                For::<EnvFiniteFinite, _>::new((*inner_agent_def).clone()),
                (&InnerEnvStructure::new(env)).into(),
                seed,
            )
            .map(|a| Box::new(a) as _),
            _ => BuildAgentFor::<EnvAnyAny, _, _>::build_agent(self, env, seed),
        }
    }
}

/// Build a multithread agent for enviornments satisfying the marker type.
pub trait BuildMultithreadAgentFor<M, OS: Space, AS: Space> {
    type MultithreadAgent: InitializeMultithreadAgent<OS::Element, AS::Element>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError>;
}

impl<M, OS, AS> BuildMultithreadAgentFor<M, OS, AS> for MultithreadAgentDef
where
    OS: RLObservationSpace,
    AS: RLActionSpace,
    AgentDef: BuildAgentFor<M, OS, AS, Agent = Box<dyn FullAgent<OS::Element, AS::Element> + Send>>,
{
    type MultithreadAgent = GenericMultithreadInitializer<OS, AS>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        use MultithreadAgentDef::*;
        Ok(match self {
            Mutex(config) => GenericMultithreadInitializer::Mutex(
                MutexAgentConfig::new(For::<M, _>::new(config.deref()))
                    .build_multithread_agent(env, seed)?,
            ),
        })
    }
}

/// Associate a type with a particular environment structure marker.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct For<M, T> {
    marker: PhantomData<fn() -> M>,
    wrapped: T,
}

impl<M, T> For<M, T> {
    // False positive in clippy or bug in cargo
    // The PhantomData of a function pointer supposedly cannot be const created
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(wrapped: T) -> Self {
        Self {
            marker: PhantomData,
            wrapped,
        }
    }
}

impl<M, T> From<T> for For<M, T> {
    fn from(wrapped: T) -> Self {
        Self::new(wrapped)
    }
}

impl<M, T> Deref for For<M, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.wrapped
    }
}

impl<M, T, OS: Space, AS: Space> BuildAgent<OS, AS> for For<M, T>
where
    T: BuildAgentFor<M, OS, AS>,
{
    type Agent = T::Agent;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        self.wrapped.build_agent(env, seed)
    }
}

impl<M, T, OS: Space, AS: Space> BuildMultithreadAgent<OS, AS> for For<M, T>
where
    T: BuildMultithreadAgentFor<M, OS, AS>,
{
    type MultithreadAgent = T::MultithreadAgent;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        self.wrapped.build_multithread_agent(env, seed)
    }
}

/// Send-able [`FullAgent`] trait object for an environment structure.
pub type DynFullAgent<OS, AS> =
    dyn FullAgent<<OS as Space>::Element, <AS as Space>::Element> + Send;

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

    fn new_worker(&mut self) -> Result<Self::Worker, BuildAgentError> {
        use GenericMultithreadInitializer::*;
        Ok(match self {
            Mutex(initializer) => Box::new(initializer.new_worker()?),
        })
    }

    fn into_manager(self) -> Self::Manager {
        use GenericMultithreadInitializer::*;
        match self {
            Mutex(initializer) => Box::new(initializer.into_manager()),
        }
    }
}
