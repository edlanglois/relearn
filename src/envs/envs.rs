use crate::spaces::Space;
use std::f32;

/// A reinforcement learning environment
///
/// Specifically, an environment instance encapsulates a specific environment state and allows
/// transitions between states.
pub trait Environment {
    type Observation;
    type Action;

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
    fn step(&mut self, action: &Self::Action) -> (Option<Self::Observation>, f32, bool);

    /// Reset the environment to an initial state.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    fn reset(&mut self) -> Self::Observation;
}

/// An environment with a well-defined external structure.
pub trait StructuredEnvironment:
    Environment<
    Observation = <<Self as StructuredEnvironment>::ObservationSpace as Space>::Element,
    Action = <<Self as StructuredEnvironment>::ActionSpace as Space>::Element,
>
{
    type ObservationSpace: Space;
    type ActionSpace: Space;

    /// Get the structure of this environment.
    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace>;
}

/// The external structure of an environment.
#[derive(Debug)]
pub struct EnvStructure<OS: Space, AS: Space> {
    /// Space containing all possible observations.
    ///
    /// This is not required to be tight:
    /// this space may contain elements that can never be produced as a state observation.
    pub observation_space: OS,
    /// The space of possible actions.
    ///
    /// Every element in this space must be a valid action.
    pub action_space: AS,
    /// A lower and upper bound on possible reward values.
    ///
    /// These bounds are not required to be tight but ideally will be as tight as possible.
    pub reward_range: (f32, f32),
    /// A discount factor applied to future rewards.
    pub discount_factor: f32,
}
