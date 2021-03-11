pub mod bandits;

pub use bandits::BernoulliBandit;

use crate::spaces::Space;
use std::f32;

/// A reinforcement learning environment
///
/// Specifically, an environment instance encapsulates a specific environment state and allows
/// transitions between states.
pub trait Environment {
    type ObservationSpace: Space;
    type ActionSpace: Space;

    /// Take a step in the environment.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    ///     Is `None` if the resulting state is terminal.
    ///     All trajectories from terminal states yield 0 reward on each step.
    /// * `reward`: The reward value for this transition
    /// * `episode_done`: Whether this step ends the episode.
    ///     - If `observation` is `None` then `episode_done` must be true.
    ///     - An episode may be done for other reasons, like a step limit.
    fn step(
        &mut self,
        action: &<<Self as Environment>::ActionSpace as Space>::Element,
    ) -> (
        Option<<<Self as Environment>::ObservationSpace as Space>::Element>,
        f32,
        bool,
    );

    /// Reset the environment to an initial state.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    fn reset(&mut self) -> <<Self as Environment>::ObservationSpace as Space>::Element;

    /// The environment observation space.
    fn observation_space(&self) -> Self::ObservationSpace;

    /// The environment action space.
    fn action_space(&self) -> Self::ActionSpace;

    /// Lower and upper bound on reward values.
    fn reward_range(&self) -> (f32, f32) {
        (-f32::INFINITY, f32::INFINITY)
    }

    /// Rate at which future rewards are discounted.
    fn discount_factor(&self) -> f32 {
        1.0
    }
}
