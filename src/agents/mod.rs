//! Reinforcement learning agents
//!
//! More agents can be found in [`crate::torch::agents`].

mod bandits;
mod batch_update;
mod finite;
pub mod history;
mod meta;
mod multithread;
mod random;
mod tabular;
#[cfg(test)]
pub mod testing;

pub use bandits::{
    BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, UCB1Agent, UCB1AgentConfig,
};
pub use batch_update::{
    BatchUpdate, BatchUpdateAgent, BatchUpdateAgentConfig, BuildBatchUpdateActor, OffPolicyAgent,
};
use finite::{BuildIndexAgent, FiniteSpaceAgent};
pub use meta::{ResettingMetaAgent, ResettingMetaAgentConfig};
pub use multithread::{MutexAgentConfig, MutexAgentManager, MutexAgentWorker};
pub use random::{RandomAgent, RandomAgentConfig};
pub use tabular::{TabularQLearningAgent, TabularQLearningAgentConfig};

use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;
use std::marker::PhantomData;
use tch::TchError;
use thiserror::Error;

/// Description of an environment step
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step<O, A> {
    /// The initial observation.
    pub observation: O,
    /// The action taken from the initial state given the initial observation.
    pub action: A,
    /// The resulting reward.
    pub reward: f64,
    /// The resulting successor state; is None if the successor state is terminal.
    /// All trajectories from a terminal state have 0 reward on each step.
    pub next_observation: Option<O>,
    /// Whether this step ends the episode.
    /// An episode is always done if it reaches a terminal state.
    /// An episode may be done for other reasons, like a step limit.
    pub episode_done: bool,
}

/// An actor that produces actions in response to a sequence of observations.
///
/// The action selection process may depend on the history of observations and actions within an
/// episode.
pub trait Actor<O, A> {
    /// Choose an action in the environment.
    ///
    /// This must be called sequentially within an episode,
    /// allowing the actor to internally maintain a history of the episode
    /// that informs the actions.
    ///
    /// # Args
    /// * `observation`: The current observation of the environment state.
    /// * `new_episode`: Whether this observation is the start of a new episode.
    fn act(&mut self, observation: &O, new_episode: bool) -> A;
}

impl<T, O, A> Actor<O, A> for Box<T>
where
    T: Actor<O, A> + ?Sized,
{
    fn act(&mut self, observation: &O, new_episode: bool) -> A {
        T::act(self, observation, new_episode)
    }
}

/// A learning agent.
///
/// Takes actions in a reinforcement learning environment and updates based on the result
/// (including reward).
pub trait Agent<O, A>: Actor<O, A> {
    /// Update the agent based on the most recent action.
    ///
    /// Must be called immediately after the corresponding call to [`Actor::act`],
    /// before any other calls to `act` are made.
    /// This allows the agent to internally cache any information used in selecting the action
    /// that would also be useful for updating on the result.
    ///
    /// # Args
    /// * `step`: The environment step resulting from the  most recent call to [`Actor::act`].
    fn update(&mut self, step: Step<O, A>, logger: &mut dyn TimeSeriesLogger);
}

impl<T, O, A> Agent<O, A> for Box<T>
where
    T: Agent<O, A> + ?Sized,
{
    fn update(&mut self, step: Step<O, A>, logger: &mut dyn TimeSeriesLogger) {
        T::update(self, step, logger)
    }
}

/// The behaviour mode of an [`Actor`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ActorMode {
    /// The actor expects to receive reward feedback on its actions.
    ///
    /// May take explicitly exploratory actions that are sub-optimal within an episode
    /// (given the agent's current knowledge) but potentially allow for better strategies to be
    /// discovered over the course of multiple episodes.
    Training,

    /// The actor does not expect reward feedback and should maximize return within each episode.
    ///
    /// The actor may learn within an episode, including taking exploratory actions that it expects
    /// to support improved performance _within that episode_.
    ///
    /// This could also be called "greedy" but that term might suggest that a history-based agent
    /// should not attempt to explore and learn within an episode.
    ///
    /// [`Agent::update`] should not be called on an agent in release mode.
    /// The agent is free to either ignore the update or perform an update.
    Release,
}

/// Supports setting the actor mode.
///
/// Unless explicitly specified during initialization,
/// actors must start in [`ActorMode::Training`] mode.
pub trait SetActorMode {
    /// Set the actor mode.
    fn set_actor_mode(&mut self, _mode: ActorMode) {
        // The default implementation just ignores the mode.
        // Many actors only have a single kind of behaviour.
    }
}

impl<T: SetActorMode + ?Sized> SetActorMode for Box<T> {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        T::set_actor_mode(self, mode)
    }
}

// TODO: Replace with BuildMultithreadAgent that constructs a manager and n workers
// Or maybe builds MultithreadAgentInitializer with the methods
// make_worker(&self) -> Self::Worker and
// into_manager(self) -> Self::Manager

pub trait BuildManagerAgent<OS: Space, AS: Space> {
    type ManagerAgent: ManagerAgent<Worker = Self::Worker>;
    type Worker: Agent<OS::Element, AS::Element> + Send + 'static;

    fn build_manager_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::ManagerAgent, BuildAgentError>;
}

/// A manager agent for a set of multi-threaded workers.
///
/// Each worker will be sent to its own thread while the manager is run on the original thread.
/// The workers will be run on a sequence of environment steps.
/// The managers and workers are responsible for internally coordinating updates and
/// synchronization.
pub trait ManagerAgent {
    type Worker: Send + 'static;

    /// Create a new worker instance.
    fn make_worker(&mut self, seed: u64) -> Self::Worker;

    /// Run the manager.
    ///
    /// This function will be run on a separate thread from the workers.
    ///
    /// For example, it might collect data from the workers, perform policy updates,
    /// and distribute the updated policy back to the workers.
    fn run(&mut self, logger: &mut dyn TimeSeriesLogger);
}

impl<T> ManagerAgent for Box<T>
where
    T: ManagerAgent + ?Sized,
{
    type Worker = T::Worker;

    fn make_worker(&mut self, seed: u64) -> Self::Worker {
        T::make_worker(self, seed)
    }

    fn run(&mut self, logger: &mut dyn TimeSeriesLogger) {
        T::run(self, logger)
    }
}

/// Wraps a [`ManagerAgent`] to return boxed workers.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BoxingManager<T, O, A> {
    inner: T,
    observation_type: PhantomData<*const O>,
    action_type: PhantomData<*const A>,
}

impl<T, O, A> BoxingManager<T, O, A> {
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            observation_type: PhantomData,
            action_type: PhantomData,
        }
    }
}

impl<T, O, A> ManagerAgent for BoxingManager<T, O, A>
where
    T: ManagerAgent,
    T::Worker: Agent<O, A>,
    O: 'static,
    A: 'static,
{
    type Worker = Box<dyn Agent<O, A> + Send>;

    fn make_worker(&mut self, seed: u64) -> Self::Worker {
        Box::new(self.inner.make_worker(seed))
    }

    fn run(&mut self, logger: &mut dyn TimeSeriesLogger) {
        self.inner.run(logger)
    }
}

pub trait FullAgent<O, A>: Agent<O, A> + SetActorMode {}
impl<O, A, T> FullAgent<O, A> for T where T: Agent<O, A> + SetActorMode + ?Sized {}

// TODO: Be more flexible about the bounds on Agent?
/// Build an agent instance for a given environment structure.
pub trait BuildAgent<OS: Space, AS: Space> {
    /// Type of agent to build
    type Agent: FullAgent<OS::Element, AS::Element>;

    /// Build an agent for the given environment structure ([`EnvStructure`]).
    ///
    /// The agent is built in [`ActorMode::Training`].
    ///
    /// # Args
    /// * `env`  - The structure of the environment in which the agent is to operate.
    /// * `seed` - A number used to seed the agent's random state,
    ///            for those agents that use deterministic pseudo-random number generation.
    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError>;
}

/// Error building an agent
#[derive(Error, Debug)]
pub enum BuildAgentError {
    #[error("space bound(s) are too loose for this agent")]
    InvalidSpaceBounds,
    #[error("reward range must not be unbounded")]
    UnboundedReward,
    #[error(transparent)]
    TorchError(#[from] TchError),
}
