//! Reinforcement learning agents
//!
//! More agents can be found in [`crate::torch::agents`].

mod bandits;
mod builder;
mod meta;
mod random;
mod tabular;
#[cfg(test)]
pub mod testing;

pub use bandits::{
    BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, UCB1Agent, UCB1AgentConfig,
};
pub use builder::{AgentBuilder, BuildAgentError};
pub use meta::ResettingMetaAgent;
pub use random::{RandomAgent, RandomAgentConfig};
pub use tabular::{TabularQLearningAgent, TabularQLearningAgentConfig};

use crate::logging::TimeSeriesLogger;

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
    /// The user should not call [`Agent::update`] on an agent in `Release` mode.
    /// The agent is free to either ignore the update or perform an update.
    Release,
}

/// Supports setting the actor mode.
pub trait SetActorMode {
    /// Set the actor mode.
    fn set_actor_mode(&mut self, _mode: ActorMode) {
        // The default implementation just ignores the mode.
        // Many actors only have a single kind of behaviour.
    }
}
