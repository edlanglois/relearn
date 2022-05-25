//! Reinforcement learning agents
//!
//! More agents can be found in [`crate::torch::agents`].

mod bandits;
pub mod buffers;
mod r#dyn;
pub mod finite;
mod meta;
mod pair;
mod random;
mod serial;
mod tabular;
#[cfg(test)]
pub mod testing;

pub use bandits::{
    BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, UCB1Agent, UCB1AgentConfig,
};
pub use buffers::{
    HistoryDataBound, WriteExperience, WriteExperienceError, WriteExperienceIncremental,
};
pub use meta::{ResettingMetaAgent, ResettingMetaAgentConfig};
pub use pair::AgentPair;
pub use r#dyn::{BoxActor, BoxAgent, DynAgent};
pub use random::{RandomAgent, RandomAgentConfig};
pub use serial::SerialActorAgent;
pub use tabular::{TabularQLearningAgent, TabularQLearningAgentConfig};

use crate::envs::EnvStructure;
use crate::logging::StatsLogger;
use crate::spaces::Space;
use crate::Prng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tch::TchError;
use thiserror::Error;

/// A reinforcement learning agent. Provides environment actors and learns from experience.
///
/// Generic over the observation type `O` and the action type `A`.
///
/// The associated `Actor` objects should be treated as though they contain references to this
/// `Agent` (this can be enforced at compile-time once [Generic Associated Types][GAT] are stable).
/// This assumption is currently enforced at runtime where relevant.
/// For example, attempting to update `Agent` may panic if any `Actor`s exist.
///
/// [GAT]: https://rust-lang.github.io/rfcs/1598-generic_associated_types.html
pub trait Agent<O, A>: BatchUpdate<O, A> {
    /// Produces actions for a sequence of environment observations.
    type Actor: Actor<O, A>;

    /// Create a new [`Actor`] with the given behaviour mode.
    ///
    /// # Implementation Note
    /// When training, new actors need to be created every time the model is changed. As such, this
    /// method should be reasonably efficient, but it is not called in such a tight loop that
    /// efficiency is critical.
    fn actor(&self, mode: ActorMode) -> Self::Actor;
}

/// Implement `Agent<O, A>` for a deref-able wrapper type generic over `T: Agent<O, A> + ?Sized`.
macro_rules! impl_wrapped_agent {
    ($wrapper:ty) => {
        impl<T, O, A> Agent<O, A> for $wrapper
        where
            T: Agent<O, A> + ?Sized,
        {
            type Actor = T::Actor;
            fn actor(&self, mode: ActorMode) -> Self::Actor {
                T::actor(self, mode)
            }
        }
    };
}
impl_wrapped_agent!(&'_ mut T);
impl_wrapped_agent!(Box<T>);

/// Take actions in an environment.
///
/// The actions may depend on the action-observation history within an episode
/// but not across episodes. This is managed with an explicit `EpisodeState` associated type.
///
/// # Design Discussion
/// ## Episode State
/// If [Generic Associated Types][GAT] were stable, an alternate strategy would be to have
/// a self-contaned `EpisodeActor<'a>` associated type with an `act(&mut self, observation: &O)`
/// method. However, this would make it challenging to store both an `Actor` and its `EpisodeActor`
/// together (if wanting a single object to act over multiple sequential episodes).
/// As such, the current `EpisodeState` strategy might still be preferable.
///
/// Another strategy (allowed without GAT) is for the `Actor` to internally manage episode state
/// and provide a `reset()` method for resetting between episodes. This lacks the benefit of being
/// able to guarantee independence between episodes via the type system.
///
/// ## Random State
/// The actor is not responsible for managing its own pseudo-random state.
/// This avoids having to frequently re-initialize the random number generator on each episode and
/// simplifies episode state definitions.
///
/// [GAT]: https://rust-lang.github.io/rfcs/1598-generic_associated_types.html
pub trait Actor<O: ?Sized, A> {
    /// Stores state for each episode.
    type EpisodeState;

    /// Create episode state for the start of a new episode.
    fn new_episode_state(&self, rng: &mut Prng) -> Self::EpisodeState;

    /// Select an action in response to an observation.
    ///
    /// May depend on and update the episode state.
    /// The observation, the selected action, and any other internal state may be stored into
    /// `episode_state`.
    fn act(&self, episode_state: &mut Self::EpisodeState, observation: &O, rng: &mut Prng) -> A;
}
/// Implement `Actor<O, A>` for a deref-able wrapper type generic over `T: Actor<O, A> + ?Sized`.
macro_rules! impl_wrapped_actor {
    ($wrapper:ty) => {
        impl<T, O, A> Actor<O, A> for $wrapper
        where
            T: Actor<O, A> + ?Sized,
            O: ?Sized,
        {
            type EpisodeState = T::EpisodeState;
            fn new_episode_state(&self, rng: &mut Prng) -> Self::EpisodeState {
                T::new_episode_state(self, rng)
            }
            fn act(
                &self,
                episode_state: &mut Self::EpisodeState,
                observation: &O,
                rng: &mut Prng,
            ) -> A {
                T::act(self, episode_state, observation, rng)
            }
        }
    };
}
impl_wrapped_actor!(&'_ T);
impl_wrapped_actor!(Box<T>);
impl_wrapped_actor!(Arc<T>);

/// Behaviour mode of an actor.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActorMode {
    /// Training mode
    ///
    /// A training-mode actor expects that its actions will be used as the basis for training data.
    /// It may take explicitly exploratory actions that are sub-optimal within an episode (given
    /// the agent's current knowledge or abilities) but that potentially allow for better
    /// strategies to be discovered when learning over the course of multiple episodes.
    Training,

    /// Evaluation mode
    ///
    /// An evaluation-mode actor attempts to maximize reward to the best of its ability in each
    /// episode. The actor may learn within an episode, including taking exploratory actions that
    /// it expects to yeild improved performace _within that episode_ but should assume that no
    /// agent updates will be performed with resulting data.
    Evaluation,
}

/// Update an agent with steps collected into history buffers.
///
/// The user is only responsible for adding data to the history buffer.
/// If old data in the buffer needs to be cleared that is the reponsibility of either the buffer
/// or the `batch_update` method.
pub trait BatchUpdate<O, A> {
    type HistoryBuffer: WriteExperience<O, A>;

    /// Create a new history buffer.
    fn buffer(&self) -> Self::HistoryBuffer;

    /// Request a minimum amount of on-policy data for the next batch update.
    fn min_update_size(&self) -> HistoryDataBound;

    /// Update the agent from a collection of history buffers.
    ///
    /// All new data inserted into the buffers since the last call must be on-policy.
    ///
    /// It is preferable but not required for the last step of the newly added data to end its
    /// episode --- either by termination or interruption. If instead the last `step.next` is
    /// `Successor::Continue` then that one step may be dropped or ignored by agents that depend
    /// on well-formed step successors.
    ///
    /// This function and the history buffer itself are jointly reponsible for managing the data
    /// within the buffer. The buffers may be emptied by the call.
    ///
    /// # Implementation Note
    /// This iterator-generic `batch_update` is necessary because some agents modify the given
    /// buffer references before passing to `batch_update` of an inner agent. This would not work
    /// if the inner agent only took a slice of buffers.
    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn StatsLogger)
    where
        Self: Sized,
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a;

    /// Update the agent from a history buffer.
    ///
    /// Like [`BatchUpdate::batch_update`] but object-safe.
    fn batch_update_single(
        &mut self,
        buffer: &mut Self::HistoryBuffer,
        logger: &mut dyn StatsLogger,
    );

    /// Update the agent from a slice of history buffers.
    ///
    /// Like [`BatchUpdate::batch_update`] but object-safe.
    fn batch_update_slice(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn StatsLogger,
    );
}

/// Implement `BatchUpdate<O, A>` for a deref-able wrapper over `T: BatchUpdate<O, A> + ?Sized`.
macro_rules! impl_wrapped_batch_update {
    ($wrapper:ty) => {
        impl<T, O, A> BatchUpdate<O, A> for $wrapper
        where
            T: BatchUpdate<O, A> + ?Sized,
        {
            type HistoryBuffer = T::HistoryBuffer;

            fn buffer(&self) -> Self::HistoryBuffer {
                T::buffer(self)
            }

            fn min_update_size(&self) -> HistoryDataBound {
                T::min_update_size(self)
            }

            fn batch_update<'a, I>(&mut self, _buffers: I, _logger: &mut dyn StatsLogger)
            where
                Self: Sized,
                I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
                Self::HistoryBuffer: 'a,
            {
                unimplemented!(
                    "batch_update not implemented for wrappers; \
                    try `batch_update_single` or `batch_update_slice`"
                )
            }
            fn batch_update_single(
                &mut self,
                buffer: &mut Self::HistoryBuffer,
                logger: &mut dyn StatsLogger,
            ) {
                T::batch_update_single(self, buffer, logger)
            }
            fn batch_update_slice(
                &mut self,
                buffers: &mut [Self::HistoryBuffer],
                logger: &mut dyn StatsLogger,
            ) {
                T::batch_update_slice(self, buffers, logger)
            }
        }
    };
}
impl_wrapped_batch_update!(&'_ mut T);
impl_wrapped_batch_update!(Box<T>);

/// Build an agent instance for a given environment structure.
pub trait BuildAgent<OS: Space, AS: Space> {
    /// Type of agent to build
    type Agent: Agent<OS::Element, AS::Element>;

    /// Build an agent for the given environment structure ([`EnvStructure`]).
    ///
    /// # Args
    /// * `env` - The structure of the environment in which the agent is to operate.
    /// * `rng` - Used for seeding the agent's pseudo-random internal parameters, if any.
    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        rng: &mut Prng,
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
