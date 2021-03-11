pub mod random;
pub mod tabular;

pub use random::RandomAgent;
pub use tabular::TabularQLearningAgent;

use crate::spaces::Space;
use rand::Rng;

/// Description of an environment step
pub struct Step<O, A> {
    /// The initial observation.
    pub observation: O,
    /// The action taken from the initial state given the initial observation.
    pub action: A,
    /// The resulting reward.
    pub reward: f32,
    /// The resulting successor state; is None if the successor state is terminal.
    /// All trajectories from a terminal state have 0 reward on each step.
    pub next_observation: Option<O>,
    /// Whether this step ends the episode.
    /// An episode is always done if it reaches a terminal state.
    /// An episode may be done for other reasons, like a step limit.
    pub episode_done: bool,
}

/// An agent that interacts with an environment
pub trait Agent<OS: Space, AS: Space> {
    /// Choose an action in the environment.
    ///
    /// This must be called sequentially within an episode.
    ///
    /// # Args
    /// * `observation`: The current observation of the environment state.
    /// * `prev_step`: The immediately preceeding environment step.
    ///     If `None` then this is the start of a new episode.
    /// * `rng`: A (pseudo) random number generator available to the agent.
    fn act<R: Rng>(
        &mut self,
        observation: &OS::Element,
        prev_step: Option<Step<OS::Element, AS::Element>>,
        rng: &mut R,
    ) -> AS::Element;
}

/// A Markov learning agent
///
/// Markov agents do not depend on history when acting,
/// except to the extent that they learn from past history.
pub trait MarkovAgent<OS: Space, AS: Space> {
    /// Choose an action for the given observation.
    fn act<R: Rng>(&self, observation: &OS::Element, rng: &mut R) -> AS::Element;

    /// Update the agent based on an environment step.
    fn update(&mut self, step: Step<OS::Element, AS::Element>);
}
