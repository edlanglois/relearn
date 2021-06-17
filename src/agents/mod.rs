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
///
/// The generated actions should not assume that any updates will follow.
/// In particular, this means that [`Actor::act`] may act more greedily compare to
/// [`Agent::act`], which assumes that updates will follow.
pub trait Actor<O, A> {
    /// Choose an action in the environment.
    ///
    /// This must be called sequentially within an episode.
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
/// Can interact with an environment and learns from the interaction.
pub trait Agent<O, A> {
    /// Choose an action in the environment.
    ///
    /// [`Agent::update`] must be called immediately after with the result.
    /// Must be called sequentially on steps within an episode.
    ///
    /// # Args
    /// * `observation`: The current observation of the environment state.
    /// * `new_episode`: Whether this observation is the start of a new episode.
    fn act(&mut self, observation: &O, new_episode: bool) -> A;

    /// Update the agent based on the most recent action.
    ///
    /// # Args
    /// * `step`: The environment step resulting from the  most recent call to [`Actor::act`].
    fn update(&mut self, step: Step<O, A>, logger: &mut dyn TimeSeriesLogger);
}

impl<T, O, A> Agent<O, A> for Box<T>
where
    T: Agent<O, A> + ?Sized,
{
    fn act(&mut self, observation: &O, new_episode: bool) -> A {
        T::act(self, observation, new_episode)
    }

    fn update(&mut self, step: Step<O, A>, logger: &mut dyn TimeSeriesLogger) {
        T::update(self, step, logger)
    }
}

/// Both an Actor and an Agent
pub trait ActorAgent<O, A>: Actor<O, A> + Agent<O, A> {}

impl<O, A, T: Actor<O, A> + Agent<O, A>> ActorAgent<O, A> for T {}
