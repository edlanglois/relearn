//! Reinforcement learning environments
mod bandits;
mod builder;
mod chain;
mod memory;
mod stateful;
#[cfg(test)]
pub mod testing;

pub use bandits::{
    Bandit, BernoulliBandit, DeterministicBandit, FixedMeansBanditConfig, PriorMeansBanditConfig,
};
pub use builder::{BuildEnvError, EnvBuilder};
pub use chain::Chain;
pub use memory::MemoryGame;
pub use stateful::{EnvWithState, WithState};

use crate::spaces::Space;
use rand::rngs::StdRng;
use std::f64;

/// A reinforcement learning environment.
///
/// This defines the environment dynamics and strucutre.
/// It does not internally manage state.
pub trait Environment {
    type State;
    type ObservationSpace: Space;
    type ActionSpace: Space;

    /// Sample a new initial state.
    fn initial_state(&self, rng: &mut StdRng) -> Self::State;

    /// Sample an observation for a state.
    fn observe(
        &self,
        state: &Self::State,
        rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element;

    /// Sample a state transition.
    ///
    /// # Returns
    /// * `state`: The resulting state.
    ///     Is `None` if the resulting state is terminal.
    ///     All trajectories from terminal states yield 0 reward on each step.
    /// * `reward`: The reward value for this transition
    /// * `episode_done`: Whether this step ends the episode.
    ///     - If `observation` is `None` then `episode_done` must be true.
    ///     - An episode may be done for other reasons, like a step limit.
    fn step(
        &self,
        state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool);

    /// The structure of this environment.
    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace>;
}

impl<E: Environment + ?Sized> Environment for Box<E> {
    type State = E::State;
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        E::initial_state(self, rng)
    }

    fn observe(
        &self,
        state: &Self::State,
        rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
        E::observe(self, state, rng)
    }

    fn step(
        &self,
        state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        E::step(self, state, action, rng)
    }

    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace> {
        E::structure(self)
    }
}

/// A reinforcement learning environment with internal state.
pub trait StatefulEnvironment {
    type ObservationSpace: Space;
    type ActionSpace: Space;

    /// Take a step in the environment.
    ///
    /// This may panic if the state has not be initialized with reset()
    /// after initialization or after a step returned `episod_done = True`.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    ///     Is `None` if the resulting state is terminal.
    ///     All trajectories from terminal states yield 0 reward on each step.
    /// * `reward`: The reward value for this transition
    /// * `episode_done`: Whether this step ends the episode.
    ///     - If `observation` is `None` then `episode_done` must be true.
    ///     - An episode may be done for other reasons, like a step limit.
    // TODO: Why not make reset() optional and have new() / step() self-reset?
    fn step(
        &mut self,
        action: &<Self::ActionSpace as Space>::Element,
    ) -> (
        Option<<Self::ObservationSpace as Space>::Element>,
        f64,
        bool,
    );

    /// Reset the environment to an initial state.
    ///
    /// Must be called before each new episode.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    fn reset(&mut self) -> <Self::ObservationSpace as Space>::Element;

    /// Get the structure of this environment.
    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace>;
}

impl<E: StatefulEnvironment + ?Sized> StatefulEnvironment for Box<E> {
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;

    fn step(
        &mut self,
        action: &<Self::ActionSpace as Space>::Element,
    ) -> (
        Option<<Self::ObservationSpace as Space>::Element>,
        f64,
        bool,
    ) {
        E::step(self, action)
    }

    fn reset(&mut self) -> <Self::ObservationSpace as Space>::Element {
        E::reset(self)
    }

    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace> {
        E::structure(self)
    }
}

/// The external structure of an environment.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct EnvStructure<OS, AS> {
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
    pub reward_range: (f64, f64),
    /// A discount factor applied to future rewards.
    pub discount_factor: f64,
}
