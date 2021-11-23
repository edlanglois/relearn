use super::{CriticDef, CriticUpdaterDef, PolicyDef, PolicyUpdaterDef};
use crate::agents::buffers::HistoryBuffer;
use crate::agents::{
    Actor, Agent, BatchUpdate, BatchUpdateAgentConfig, BetaThompsonSamplingAgentConfig,
    BoxingMultithreadInitializer, BuildAgent, BuildAgentError, BuildBatchUpdateActor,
    BuildMultithreadAgent, FullAgent, InitializeMultithreadAgent, MultithreadAgentManager,
    MultithreadBatchAgentConfig, MutexAgentConfig, RandomAgentConfig, ResettingMetaAgent,
    SetActorMode, SyncParamsAny, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::envs::{EnvStructure, InnerEnvStructure, MetaObservationSpace};
use crate::logging::{Loggable, TimeSeriesLogger};
use crate::spaces::{
    ElementRefInto, EncoderFeatureSpace, FiniteSpace, NonEmptySpace,
    ParameterizedDistributionSpace, SampleSpace, Space,
};
use crate::torch::agents::ActorCriticConfig;
use crate::utils::any::AsAny;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use tch::Tensor;

/// Agents implementing both [`Agent`] and [`BatchUpdate`](crate::agents::BatchUpdate)
#[derive(Debug, Clone, PartialEq)]
pub enum OptionalBatchAgentDef {
    /// Selects actions randomly without learning.
    Random,
    /// Epsilon-greedy tabular Q learning.
    TabularQLearning(TabularQLearningAgentConfig),
    /// Thompson sampling of for Bernoulli rewards using Beta priors.
    ///
    /// Assumes no relationship between states.
    BetaThompsonSampling(BetaThompsonSamplingAgentConfig),
    /// UCB1 agent from Auer 2002
    UCB1(UCB1AgentConfig),
}

/// Agent definition
#[derive(Debug, Clone, PartialEq)]
pub enum AgentDef {
    /// Pass-through for [`OptionalBatchAgentDef`] agents viewed as regular synchronous agents.
    AsSync(OptionalBatchAgentDef),
    /// [`BatchUpdateAgent`](crate::agents::BatchUpdateAgent) for [`BatchActorDef`] agents.
    Batch(BatchUpdateAgentConfig<BatchActorDef>),
    /// Applies a non-meta agent to a meta environment by resetting between trials
    ResettingMeta(Box<AgentDef>),
}

/// Actors implementing [`BatchUpdate`](crate::agents::BatchUpdate).
#[derive(Debug, Clone, PartialEq)]
pub enum BatchActorDef {
    /// Pass-through for [`OptionalBatchAgentDef`] agents.
    AsBatch(OptionalBatchAgentDef),
    /// Torch actor-critic agent
    ActorCritic(Box<ActorCriticConfig<PolicyDef, PolicyUpdaterDef, CriticDef, CriticUpdaterDef>>),
}

/// Multithread agent definition
#[derive(Debug, Clone, PartialEq)]
pub enum MultithreadAgentDef {
    /// A mutex-based simulated multithread agent. Does not provide meaningful parallelism.
    Mutex(AgentDef),
    /// Multithread agent with batch updates performed by the manger from data sent by workers.
    Batch(MultithreadBatchAgentConfig<BatchActorDef>),
}

/// A comprehensive space trait for use by RL agents.
///
/// This includes most interfaces required by any agent, environment, or simulator
/// excluding interfaces that can only apply to some spaces, like [`FiniteSpace`].
pub trait RLSpace:
    Space
    + NonEmptySpace
    + SampleSpace
    + ElementRefInto<Loggable>
    + Debug
    + Clone
    + Send
    + Sync
    + 'static
{
}
impl<
        T: Space
            + NonEmptySpace
            + SampleSpace
            + ElementRefInto<Loggable>
            + Debug
            + Clone
            + Send
            + Sync
            + 'static,
    > RLSpace for T
{
}

/// Comprehensive observation space for use in reinforcement learning
pub trait RLObservationSpace: RLSpace + EncoderFeatureSpace {}
impl<T: RLSpace + EncoderFeatureSpace> RLObservationSpace for T {}

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

// == EnvAnyAny ==

impl<OS, AS> BuildAgentFor<EnvAnyAny, OS, AS> for OptionalBatchAgentDef
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
        use OptionalBatchAgentDef::*;
        match self {
            Random => RandomAgentConfig::new()
                .build_agent(env, seed)
                .map(|a| Box::new(a) as _),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
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
            AsSync(agent_def) => {
                BuildAgentFor::<EnvAnyAny, _, _>::build_agent(agent_def, env, seed)
            }
            Batch(config) => BatchUpdateAgentConfig::new(
                For::<EnvAnyAny, _>::new(&config.actor_config),
                config.history_buffer_config,
            )
            .build_agent(env, seed)
            .map(|a| Box::new(a) as _),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
    }
}

// == EnvFiniteFinite ==

impl<OS, AS> BuildAgentFor<EnvFiniteFinite, OS, AS> for OptionalBatchAgentDef
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
        use OptionalBatchAgentDef::*;
        match self {
            TabularQLearning(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            BetaThompsonSampling(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            UCB1(config) => config.build_agent(env, seed).map(|a| Box::new(a) as _),
            _ => BuildAgentFor::<EnvAnyAny, _, _>::build_agent(self, env, seed),
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
            AsSync(agent_def) => {
                BuildAgentFor::<EnvFiniteFinite, _, _>::build_agent(agent_def, env, seed)
            }
            Batch(config) => BatchUpdateAgentConfig::new(
                For::<EnvFiniteFinite, _>::new(&config.actor_config),
                config.history_buffer_config,
            )
            .build_agent(env, seed)
            .map(|a| Box::new(a) as _),
            _ => BuildAgentFor::<EnvAnyAny, _, _>::build_agent(self, env, seed),
        }
    }
}

// == EnvMetaFiniteFinite ==

impl<OS, AS> BuildAgentFor<EnvMetaFiniteFinite, MetaObservationSpace<OS, AS>, AS>
    for OptionalBatchAgentDef
where
    OS: RLObservationSpace + FiniteSpace,
    AS: RLActionSpace + FiniteSpace,
    // I think this ought to be inferable but for some reason it isn't
    MetaObservationSpace<OS, AS>: RLObservationSpace,
{
    type Agent = Box<DynFullAgent<MetaObservationSpace<OS, AS>, AS>>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        BuildAgentFor::<EnvAnyAny, _, _>::build_agent(self, env, seed)
    }
}

impl<OS, AS> BuildAgentFor<EnvMetaFiniteFinite, MetaObservationSpace<OS, AS>, AS> for AgentDef
where
    OS: RLObservationSpace + FiniteSpace,
    AS: RLActionSpace + FiniteSpace,
    // I think this ought to be inferable but for some reason it isn't
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
            AsSync(agent_def) => {
                BuildAgentFor::<EnvMetaFiniteFinite, _, _>::build_agent(agent_def, env, seed)
            }
            Batch(config) => BatchUpdateAgentConfig::new(
                For::<EnvMetaFiniteFinite, _>::new(&config.actor_config),
                config.history_buffer_config,
            )
            .build_agent(env, seed)
            .map(|a| Box::new(a) as _),
            ResettingMeta(inner_agent_def) => ResettingMetaAgent::new(
                For::<EnvFiniteFinite, _>::new((*inner_agent_def).clone()),
                (&InnerEnvStructure::new(env)).into(),
                seed,
            )
            .map(|a| Box::new(a) as _),
            // _ => BuildAgentFor::<EnvAnyAny, _, _>::build_agent(self, env, seed),
        }
    }
}

/// Build a batch update agent for environments satisfying the marker type.
pub trait BuildBatchUpdateActorFor<M, OS: Space, AS: Space> {
    type BatchUpdateActor: FullBatchUpdateActor<OS::Element, AS::Element>;

    /// See [`BuildBatchUpdateActor::build_batch_update_actor`].
    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError>;
}

impl<T, M, OS, AS> BuildBatchUpdateActorFor<M, OS, AS> for T
where
    T: Deref + ?Sized,
    T::Target: BuildBatchUpdateActorFor<M, OS, AS>,
    OS: Space,
    AS: Space,
{
    type BatchUpdateActor = <T::Target as BuildBatchUpdateActorFor<M, OS, AS>>::BatchUpdateActor;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        self.deref().build_batch_update_actor(env, seed)
    }
}

impl<OS, AS> BuildBatchUpdateActorFor<EnvAnyAny, OS, AS> for OptionalBatchAgentDef
where
    OS: RLObservationSpace,
    AS: RLActionSpace,
{
    type BatchUpdateActor = Box<DynFullBatchUpdateActor<OS, AS>>;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        use OptionalBatchAgentDef::*;
        match self {
            Random => RandomAgentConfig::new()
                .build_agent(env, seed)
                .map(|a| Box::new(a) as _),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
    }
}

impl<OS, AS> BuildBatchUpdateActorFor<EnvFiniteFinite, OS, AS> for OptionalBatchAgentDef
where
    OS: RLObservationSpace + FiniteSpace,
    AS: RLActionSpace + FiniteSpace,
{
    type BatchUpdateActor = Box<DynFullBatchUpdateActor<OS, AS>>;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        use OptionalBatchAgentDef::*;
        match self {
            TabularQLearning(config) => config
                .build_batch_update_actor(env, seed)
                .map(|a| Box::new(a) as _),
            BetaThompsonSampling(config) => config
                .build_batch_update_actor(env, seed)
                .map(|a| Box::new(a) as _),
            UCB1(config) => config
                .build_batch_update_actor(env, seed)
                .map(|a| Box::new(a) as _),
            _ => BuildBatchUpdateActorFor::<EnvAnyAny, _, _>::build_batch_update_actor(
                self, env, seed,
            ),
        }
    }
}

impl<OS, AS> BuildBatchUpdateActorFor<EnvMetaFiniteFinite, MetaObservationSpace<OS, AS>, AS>
    for OptionalBatchAgentDef
where
    OS: RLObservationSpace + FiniteSpace,
    AS: RLActionSpace + FiniteSpace,
    // I think this ought to be inferable but for some reason it isn't
    MetaObservationSpace<OS, AS>: RLObservationSpace,
{
    type BatchUpdateActor = Box<DynFullBatchUpdateActor<MetaObservationSpace<OS, AS>, AS>>;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        BuildBatchUpdateActorFor::<EnvAnyAny, _, _>::build_batch_update_actor(self, env, seed)
    }
}

impl<M, OS, AS> BuildBatchUpdateActorFor<M, OS, AS> for BatchActorDef
where
    OS: RLObservationSpace,
    AS: RLActionSpace,
    OptionalBatchAgentDef: BuildBatchUpdateActorFor<
        M,
        OS,
        AS,
        BatchUpdateActor = Box<DynFullBatchUpdateActor<OS, AS>>,
    >,
{
    type BatchUpdateActor = Box<DynFullBatchUpdateActor<OS, AS>>;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        use BatchActorDef::*;
        match self {
            AsBatch(agent_def) => BuildBatchUpdateActorFor::<M, OS, AS>::build_batch_update_actor(
                agent_def, env, seed,
            ),
            ActorCritic(config) => config
                .as_ref()
                .build_batch_update_actor(env, seed)
                .map(|a| Box::new(a) as _),
        }
    }
}

/// Build a multithread agent for environments satisfying the marker type.
pub trait BuildMultithreadAgentFor<M, OS: Space, AS: Space> {
    type MultithreadAgent: InitializeMultithreadAgent<OS::Element, AS::Element>;

    /// See [`BuildMultithreadAgent::build_multithread_agent`].
    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError>;
}

impl<M, OS, AS> BuildMultithreadAgentFor<M, OS, AS> for MultithreadAgentDef
where
    M: Clone + 'static,
    OS: RLObservationSpace,
    AS: RLActionSpace,
    AgentDef: BuildAgentFor<M, OS, AS, Agent = Box<dyn FullAgent<OS::Element, AS::Element> + Send>>,
    OptionalBatchAgentDef: BuildBatchUpdateActorFor<
        M,
        OS,
        AS,
        BatchUpdateActor = Box<DynFullBatchUpdateActor<OS, AS>>,
    >,
{
    type MultithreadAgent = Box<DynInitializeMultithreadAgent<OS, AS>>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        use MultithreadAgentDef::*;
        Ok(match self {
            Mutex(config) => Box::new(BoxingMultithreadInitializer::new(
                MutexAgentConfig::new(For::<M, _>::new(config))
                    .build_multithread_agent(env, seed)?,
            )),
            Batch(config) => match &config.actor_config {
                BatchActorDef::AsBatch(optional_batch_agent_def) => {
                    Box::new(BoxingMultithreadInitializer::new(
                        MultithreadBatchAgentConfig {
                            // TODO: Is this clone necessary?
                            actor_config: For::<M, _>::new(optional_batch_agent_def.clone()),
                            buffer_config: config.buffer_config,
                        }
                        .build_multithread_agent(env, seed)?,
                    ))
                }
                BatchActorDef::ActorCritic(_) => todo!(),
            },
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

impl<M, T, OS: Space, AS: Space> BuildBatchUpdateActor<OS, AS> for For<M, T>
where
    T: BuildBatchUpdateActorFor<M, OS, AS>,
{
    type BatchUpdateActor = T::BatchUpdateActor;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        self.wrapped.build_batch_update_actor(env, seed)
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

pub trait FullBatchUpdateActor<O, A>:
    Actor<O, A> + BatchUpdate<O, A> + SyncParamsAny + AsAny + SetActorMode
{
}
impl<O, A, T> FullBatchUpdateActor<O, A> for T where
    T: Actor<O, A> + BatchUpdate<O, A> + SyncParamsAny + AsAny + SetActorMode + ?Sized
{
}

// TODO: Move to src/agents/batch.rs? Exporting macros is a pain
//
// A generic implementation `BatchUpdate<O, A> for Box<T>` conflicts with the
// generic implementation below of `BatchUpdate<O, A> for T where T: OffPolicyAgent + Agent<O, A>`
// because "downstream crates may implement trait `OffPolicyAgent` for type `Box<_>`
macro_rules! box_impl_batch_update {
    ($type:ty) => {
        impl<O, A> BatchUpdate<O, A> for Box<$type> {
            fn batch_update(
                &mut self,
                history: &dyn HistoryBuffer<O, A>,
                logger: &mut dyn TimeSeriesLogger,
            ) {
                self.as_mut().batch_update(history, logger)
            }
        }
    };
}
box_impl_batch_update!(dyn FullBatchUpdateActor<O, A> + Send); // TODO: + Sync

/// `Send` and `Sync` [`FullBatchUpdateActor`] trait object for an environment structure.
pub type DynFullBatchUpdateActor<OS, AS> =
    dyn FullBatchUpdateActor<<OS as Space>::Element, <AS as Space>::Element> + Send; // TODO: +Sync

/// [`InitializeMultithreadAgent`] trait object with boxed manager and workers.
pub type DynInitializeMultithreadAgent<OS, AS> = dyn InitializeMultithreadAgent<
    <OS as Space>::Element,
    <AS as Space>::Element,
    Manager = Box<dyn MultithreadAgentManager>,
    Worker = Box<dyn Agent<<OS as Space>::Element, <AS as Space>::Element> + Send>,
>;
