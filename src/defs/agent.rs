use super::{CriticDef, CriticUpdaterDef, PolicyUpdaterDef, SeqModDef};
use crate::agents::{
    Agent, AgentBuilder, BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, BoxingManager,
    BuildAgentError, ManagerAgent, MutexAgentManager, RandomAgent, RandomAgentConfig,
    ResettingMetaAgent, TabularQLearningAgent, TabularQLearningAgentConfig, UCB1Agent,
    UCB1AgentConfig,
};
use crate::envs::{EnvStructure, InnerEnvStructure, MetaObservationSpace};
use crate::logging::Loggable;
use crate::spaces::{
    BatchFeatureSpace, ElementRefInto, FeatureSpace, FiniteSpace, ParameterizedDistributionSpace,
    SampleSpace, Space,
};
use crate::torch::agents::{ActorCriticBoxedAgent, ActorCriticConfig};
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
    /// Torch actor-critic agent
    ActorCritic(Box<ActorCriticConfig<SeqModDef, PolicyUpdaterDef, CriticDef, CriticUpdaterDef>>),
    /// Applies a non-meta agent to a meta environment by resetting between trials
    ResettingMeta(Box<AgentDef>),
}

/// Multi-thread agent definition
#[derive(Debug, Clone, PartialEq)]
pub enum MultiThreadAgentDef {
    /// A mutex-based simulated multi-thread agent. Does not provide meaningful parallelism.
    Mutex(Box<AgentDef>),
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

macro_rules! agent_builder_boxed_for_any_any {
    ($type:ty$(, $bound:tt )*) => {
        impl<T, E> AgentBuilder<Box<$type>, E> for ForAnyAny<T>
        where
            T: Borrow<AgentDef>,
            E: EnvStructure + ?Sized,
            <E as EnvStructure>::ObservationSpace: RLObservationSpace $(+ $bound)* + 'static,
            <E as EnvStructure>::ActionSpace: RLActionSpace $(+ $bound)* + 'static,
        {
            fn build_agent(&self, env: &E, seed: u64) -> Result<Box<$type>, BuildAgentError> {
                use AgentDef::*;
                match self.0.borrow() {
                    Random => RandomAgentConfig::new()
                        .build_agent(env, seed)
                        .map(|a: RandomAgent<_>| Box::new(a) as _),
                    ActorCritic(config) => config
                        .as_ref()
                        .build_agent(env, seed)
                        .map(|a: ActorCriticBoxedAgent<_, _>| Box::new(a) as _),
                    _ => Err(BuildAgentError::InvalidSpaceBounds),
                }
            }
        }
    };
}
agent_builder_boxed_for_any_any!(DynEnvAgent<E>);
// agent_builder_boxed_for_any_any!(DynSendEnvAgent<E>, Send);

impl<T, E> AgentBuilder<Box<DynSendEnvAgent<E>>, E> for ForAnyAny<T>
where
    T: Borrow<AgentDef>,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: RLObservationSpace + Send + 'static,
    <E as EnvStructure>::ActionSpace: RLActionSpace + Send + 'static,
{
    fn build_agent(&self, env: &E, seed: u64) -> Result<Box<DynSendEnvAgent<E>>, BuildAgentError> {
        use AgentDef::*;
        match self.0.borrow() {
            Random => RandomAgentConfig::new()
                .build_agent(env, seed)
                .map(|a: RandomAgent<_>| Box::new(a) as _),
            // TODO: Fix and use macro
            ActorCritic(_) => panic!("ActorCritic is not yet Send"),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
    }
}

impl<T, E> AgentBuilder<Box<DynEnvManagerAgent<E>>, E> for ForAnyAny<T>
where
    T: Borrow<MultiThreadAgentDef>,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: RLObservationSpace + Send + 'static,
    <E as EnvStructure>::ActionSpace: RLActionSpace + Send + 'static,
{
    fn build_agent(
        &self,
        env: &E,
        seed: u64,
    ) -> Result<Box<DynEnvManagerAgent<E>>, BuildAgentError> {
        use MultiThreadAgentDef::*;
        match self.0.borrow() {
            Mutex(config) => ForAnyAny(config.borrow()).build_agent(env, seed).map(
                |a: Box<DynSendEnvAgent<E>>| {
                    Box::new(BoxingManager::new(MutexAgentManager::new(a))) as _
                },
            ),
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

macro_rules! agent_builder_boxed_for_finite_finite {
    ($type:ty$(, $bound:tt )*) => {
        impl<T, E> AgentBuilder<Box<$type>, E> for ForFiniteFinite<T>
        where
            T: Borrow<AgentDef>,
            E: EnvStructure + ?Sized,
            <E as EnvStructure>::ObservationSpace: RLObservationSpace + FiniteSpace $(+ $bound )* + 'static,
            <E as EnvStructure>::ActionSpace: RLActionSpace + FiniteSpace $(+ $bound )* + 'static,
        {
            fn build_agent(&self, env: &E, seed: u64) -> Result<Box<$type>, BuildAgentError> {
                use AgentDef::*;
                match self.0.borrow() {
                    TabularQLearning(config) => config
                        .build_agent(env, seed)
                        .map(|a: TabularQLearningAgent<_, _>| Box::new(a) as _),
                    BetaThompsonSampling(config) => config
                        .build_agent(env, seed)
                        .map(|a: BetaThompsonSamplingAgent<_, _>| Box::new(a) as _),
                    UCB1(config) => config
                        .build_agent(env, seed)
                        .map(|a: UCB1Agent<_, _>| Box::new(a) as _),
                    agent_def => ForAnyAny::new(agent_def).build_agent(env, seed),
                }
            }
        }
    };
}

agent_builder_boxed_for_finite_finite!(DynEnvAgent<E>);
agent_builder_boxed_for_finite_finite!(DynSendEnvAgent<E>, Send);

impl<T, E> AgentBuilder<Box<DynEnvManagerAgent<E>>, E> for ForFiniteFinite<T>
where
    T: Borrow<MultiThreadAgentDef>,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: RLObservationSpace + FiniteSpace + Send + 'static,
    <E as EnvStructure>::ActionSpace: RLActionSpace + FiniteSpace + Send + 'static,
{
    fn build_agent(
        &self,
        env: &E,
        seed: u64,
    ) -> Result<Box<DynEnvManagerAgent<E>>, BuildAgentError> {
        use MultiThreadAgentDef::*;
        match self.0.borrow() {
            Mutex(config) => ForFiniteFinite(config.borrow()).build_agent(env, seed).map(
                |a: Box<DynSendEnvAgent<E>>| {
                    Box::new(BoxingManager::new(MutexAgentManager::new(a))) as _
                },
            ),
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

macro_rules! agent_builder_boxed_for_meta_finite_finite {
    ($type:ty$(, $bound:tt )*) => {
        impl<T, E, OS, AS> AgentBuilder<Box<$type>, E> for ForMetaFiniteFinite<T>
        where
            T: Borrow<AgentDef>,
            E: EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS> + ?Sized,
            <E as EnvStructure>::ObservationSpace: RLObservationSpace + 'static,
            OS: RLObservationSpace + FiniteSpace + Clone $(+ $bound)* + 'static,
            // ResettingMetaAgent: Send requires OS::Element: Send
            <OS as Space>::Element: Clone $(+ $bound)*,
            AS: RLActionSpace + FiniteSpace + Clone $(+ $bound)* + 'static,
            <AS as Space>::Element: Clone,
        {
            fn build_agent(&self, env: &E, seed: u64) -> Result<Box<$type>, BuildAgentError> {
                use AgentDef::*;

                match self.0.borrow() {
                    ResettingMeta(inner_agent_def) => {
                        Ok(Box::new(ResettingMetaAgent::<
                                    _,
                                    Box<dyn Agent<OS::Element, AS::Element> $(+ $bound)*>,
                                    _, _>::new(
                            ForFiniteFinite::new(inner_agent_def.as_ref().clone()),
                            (&InnerEnvStructure::<E, &E>::new(env)).into(),
                            seed,
                        )) as _)
                    }
                    agent_def => ForAnyAny::new(agent_def).build_agent(env, seed),
                }
            }
        }
    };
}

agent_builder_boxed_for_meta_finite_finite!(DynEnvAgent<E>);
agent_builder_boxed_for_meta_finite_finite!(DynSendEnvAgent<E>, Send);

impl<T, E, OS, AS> AgentBuilder<Box<DynEnvManagerAgent<E>>, E> for ForMetaFiniteFinite<T>
where
    T: Borrow<MultiThreadAgentDef>,
    E: EnvStructure<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS> + ?Sized,
    <E as EnvStructure>::ObservationSpace: RLObservationSpace + 'static,
    OS: RLObservationSpace + FiniteSpace + Clone + Send + 'static,
    <OS as Space>::Element: Clone + Send,
    AS: RLActionSpace + FiniteSpace + Clone + Send + 'static,
    <AS as Space>::Element: Clone,
{
    fn build_agent(
        &self,
        env: &E,
        seed: u64,
    ) -> Result<Box<DynEnvManagerAgent<E>>, BuildAgentError> {
        use MultiThreadAgentDef::*;
        match self.0.borrow() {
            Mutex(config) => ForMetaFiniteFinite(config.borrow())
                .build_agent(env, seed)
                .map(|a: Box<DynSendEnvAgent<E>>| {
                    Box::new(BoxingManager::new(MutexAgentManager::new(a))) as _
                }),
        }
    }
}

/// The agent trait object for a given environment structure.
pub type DynEnvAgent<E> = dyn Agent<
    <<E as EnvStructure>::ObservationSpace as Space>::Element,
    <<E as EnvStructure>::ActionSpace as Space>::Element,
>;

/// The send-able trait object ofr a given environment structure.
pub type DynSendEnvAgent<E> = dyn Agent<
        <<E as EnvStructure>::ObservationSpace as Space>::Element,
        <<E as EnvStructure>::ActionSpace as Space>::Element,
    > + Send;

/// The agent manager trait object for a given environment structure.
///
/// See also [`BoxingManager`](crate::agents::BoxingManager).
pub type DynEnvManagerAgent<E> = dyn ManagerAgent<Worker = Box<DynSendEnvAgent<E>>>;
