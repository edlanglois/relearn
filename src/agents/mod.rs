//! Reinforcement learning agents
//!
//! More agents can be found in [`crate::torch::agents`].

mod bandits;
mod batch;
pub mod buffers;
mod finite;
mod meta;
mod pair;
mod random;
mod stateful;
mod tabular;
#[cfg(test)]
pub mod testing;

pub use bandits::{
    BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, UCB1Agent, UCB1AgentConfig,
};
pub use batch::{AsyncAgent, BatchedUpdates, SerialBatchAgent, SerialBatchConfig};
pub use buffers::{HistoryBuffer, WriteHistoryBuffer};
use finite::{BuildIndexAgent, FiniteSpaceAgent};
pub use meta::{ResettingMetaAgent, ResettingMetaAgentConfig};
pub use random::{RandomAgent, RandomAgentConfig};
pub use stateful::{PureAsActor, PureAsActorConfig};
pub use tabular::{TabularQLearningAgent, TabularQLearningAgentConfig};

use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::simulation::TransientStep;
use crate::spaces::Space;
use tch::TchError;
use thiserror::Error;

/// An actor with explicit episodic state.
pub trait PureActor<O, A> {
    type State;

    /// Create a new episodic state object.
    fn initial_state(&self, seed: u64) -> Self::State;

    /// Choose an action in the environment.
    fn act(&self, state: &mut Self::State, observation: &O) -> A;
}

impl<T, O, A> PureActor<O, A> for &'_ T
where
    T: PureActor<O, A> + ?Sized,
{
    type State = T::State;
    fn initial_state(&self, seed: u64) -> Self::State {
        T::initial_state(self, seed)
    }
    fn act(&self, state: &mut Self::State, observation: &O) -> A {
        T::act(self, state, observation)
    }
}

impl<T, O, A> PureActor<O, A> for Box<T>
where
    T: PureActor<O, A> + ?Sized,
{
    type State = T::State;
    fn initial_state(&self, seed: u64) -> Self::State {
        T::initial_state(self, seed)
    }
    fn act(&self, state: &mut Self::State, observation: &O) -> A {
        T::act(self, state, observation)
    }
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
    /// that informs its actions.
    fn act(&mut self, observation: &O) -> A;

    /// Reset the actor for a new episode.
    fn reset(&mut self);
}

impl<T, O, A> Actor<O, A> for &'_ mut T
where
    T: Actor<O, A> + ?Sized,
{
    fn act(&mut self, observation: &O) -> A {
        T::act(self, observation)
    }

    fn reset(&mut self) {
        T::reset(self)
    }
}

impl<T, O, A> Actor<O, A> for Box<T>
where
    T: Actor<O, A> + ?Sized,
{
    fn act(&mut self, observation: &O) -> A {
        T::act(self, observation)
    }

    fn reset(&mut self) {
        T::reset(self)
    }
}

/// Update an agent with the immediate result of an action (synchronous update).
pub trait SynchronousUpdate<O, A> {
    /// Update the agent based on the most recent action.
    ///
    /// Must be called immediately after the corresponding call to `act`,
    /// before any other calls to the agent's methods.
    /// This allows the agent to internally cache any information used in selecting the action
    /// that would also be useful for updating on the result.
    fn update(&mut self, step: TransientStep<O, A>, logger: &mut dyn TimeSeriesLogger);
}

impl<T, O, A> SynchronousUpdate<O, A> for &'_ mut T
where
    T: SynchronousUpdate<O, A> + ?Sized,
{
    fn update(&mut self, step: TransientStep<O, A>, logger: &mut dyn TimeSeriesLogger) {
        T::update(self, step, logger)
    }
}

impl<T, O, A> SynchronousUpdate<O, A> for Box<T>
where
    T: SynchronousUpdate<O, A> + ?Sized,
{
    fn update(&mut self, step: TransientStep<O, A>, logger: &mut dyn TimeSeriesLogger) {
        T::update(self, step, logger)
    }
}

/// Update an agent with steps collected into history buffers.
pub trait BatchUpdate<O, A> {
    type HistoryBuffer: WriteHistoryBuffer<O, A>;

    /// Update the agent from a collection of history buffers.
    ///
    /// This function is responsible for draining the buffer if the data should not be reused.
    /// Any data left in the buffer should remain for the next call to `batch_update`.
    ///
    /// All new data inserted into the buffers since the last call must be on-policy.
    fn batch_update(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn TimeSeriesLogger,
    );
}

impl<T, O, A> BatchUpdate<O, A> for &'_ mut T
where
    T: BatchUpdate<O, A>,
{
    type HistoryBuffer = T::HistoryBuffer;
    fn batch_update(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn TimeSeriesLogger,
    ) {
        T::batch_update(self, buffers, logger)
    }
}
impl<T, O, A> BatchUpdate<O, A> for Box<T>
where
    T: BatchUpdate<O, A>,
{
    type HistoryBuffer = T::HistoryBuffer;
    fn batch_update(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn TimeSeriesLogger,
    ) {
        T::batch_update(self, buffers, logger)
    }
}

/// Make a new actor instance with the same model.
///
/// The new actor can include references to self.
pub trait MakeActor<'a, O, A> {
    type Actor: Actor<O, A> + Send;
    fn make_actor(&'a self, seed: u64) -> Self::Actor;
}

pub trait BuildBatchAgent<OS: Space, AS: Space> {
    type HistoryBuffer: WriteHistoryBuffer<OS::Element, AS::Element> + Send;
    type BatchAgent: BatchUpdate<OS::Element, AS::Element, HistoryBuffer = Self::HistoryBuffer>
        + SetActorMode;

    /// Build a new history buffer.
    fn build_buffer(&self) -> Self::HistoryBuffer;

    /// Build a new batch agent for the given environment structure ([`EnvStructure`]).
    ///
    /// The agent is built in [`ActorMode::Training`].
    fn build_batch_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchAgent, BuildAgentError>;
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

impl<T: SetActorMode + ?Sized> SetActorMode for &'_ mut T {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        T::set_actor_mode(self, mode)
    }
}

impl<T: SetActorMode + ?Sized> SetActorMode for Box<T> {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        T::set_actor_mode(self, mode)
    }
}

/// Build an agent instance for a given environment structure.
pub trait BuildAgent<OS: Space, AS: Space> {
    /// Type of agent to build
    type Agent: SynchronousUpdate<OS::Element, AS::Element> + SetActorMode;

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
