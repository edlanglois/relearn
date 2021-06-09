//! Reinforcement learning environments
mod bandits;
mod builder;
mod chain;
mod mdps;
mod memory;
mod meta;
mod stateful;
mod step_limit;
#[cfg(test)]
pub mod testing;

pub use bandits::{
    Bandit, BernoulliBandit, DeterministicBandit, FixedMeansBanditConfig, OneHotBandits,
    PriorMeansBanditConfig, UniformBernoulliBandits,
};
pub use builder::{BuildEnvError, EnvBuilder, EnvDistBuilder};
pub use chain::Chain;
pub use memory::MemoryGame;
pub use meta::{
    InnerEnvStructure, MetaEnv, MetaEnvConfig, MetaEnvState, MetaObservationSpace, StatefulMetaEnv,
};
pub use stateful::{DistWithState, EnvWithState, IntoStateful};
pub use step_limit::StepLimit;

use crate::spaces::Space;
use rand::rngs::StdRng;
use std::f64;

/// The external structure of a reinforcement learning environment.
pub trait EnvStructure {
    type ObservationSpace: Space;
    type ActionSpace: Space;

    /// Space containing all possible observations.
    ///
    /// This is not required to be tight:
    /// the space may contain elements that can never be produced as a state observation.
    fn observation_space(&self) -> Self::ObservationSpace;

    /// The space of all possible actions.
    ///
    /// Every element in this space must be a valid action.
    fn action_space(&self) -> Self::ActionSpace;

    /// A lower and upper bound on possible reward values.
    ///
    /// These bounds are not required to be tight but ideally will be as tight as possible.
    fn reward_range(&self) -> (f64, f64);

    /// A discount factor applied to future rewards.
    ///
    /// A value between `0` and `1`, inclusive.
    fn discount_factor(&self) -> f64;
}

impl<E: EnvStructure + ?Sized> EnvStructure for Box<E> {
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        E::observation_space(self)
    }
    fn action_space(&self) -> Self::ActionSpace {
        E::action_space(self)
    }
    fn reward_range(&self) -> (f64, f64) {
        E::reward_range(self)
    }
    fn discount_factor(&self) -> f64 {
        E::discount_factor(self)
    }
}

/// Stored copy of an environment structure.
///
/// See [`EnvStructure`] for details.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoredEnvStructure<OS: Space, AS: Space> {
    pub observation_space: OS,
    pub action_space: AS,
    pub reward_range: (f64, f64),
    pub discount_factor: f64,
}

impl<OS, AS> EnvStructure for StoredEnvStructure<OS, AS>
where
    OS: Space + Clone,
    AS: Space + Clone,
{
    type ObservationSpace = OS;
    type ActionSpace = AS;
    fn observation_space(&self) -> Self::ObservationSpace {
        self.observation_space.clone()
    }
    fn action_space(&self) -> Self::ActionSpace {
        self.action_space.clone()
    }
    fn reward_range(&self) -> (f64, f64) {
        self.reward_range
    }
    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }
}

impl<E> From<&E> for StoredEnvStructure<E::ObservationSpace, E::ActionSpace>
where
    E: EnvStructure + ?Sized,
{
    fn from(env: &E) -> Self {
        Self {
            observation_space: env.observation_space(),
            action_space: env.action_space(),
            reward_range: env.reward_range(),
            discount_factor: env.discount_factor(),
        }
    }
}

/// A reinforcement learning environment.
///
/// This defines the environment dynamics and strucutre.
/// It does not internally manage state.
pub trait Environment: EnvStructure {
    type State;

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
}

impl<E: Environment + ?Sized> Environment for Box<E> {
    type State = E::State;

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
}

/// A reinforcement learning environment with internal state.
///
/// Prefer implementing [`Environment`] since [`EnvWithState`] can be used
/// to create a `StatefulEnvironment` out of an `Environment`.
pub trait StatefulEnvironment: EnvStructure {
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
}

impl<E: StatefulEnvironment + ?Sized> StatefulEnvironment for Box<E> {
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
}

/// A distribution of envrionments sharing the same structure.
///
/// The spaces / intervals of each sampled environment must be equal to
/// or a subset of the spaces for `EnvDistribution`.
/// The discount factor of the sampled environments must be the same.
pub trait EnvDistribution: EnvStructure {
    type Environment: EnvStructure<
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
    >;

    /// Sample an environment from the distribution.
    ///
    /// # Args
    /// * `rng` - Random number generator used for sampling the environment structure and for
    ///           seeding any internal randomness of the environment dynamics.
    fn sample_environment(&self, rng: &mut StdRng) -> Self::Environment;

    /// Convert a distribution of stateless envs. to a distribution of stateful envs.
    fn into_stateful(self) -> DistWithState<Self>
    where
        Self: Sized,
        Self::Environment: Environment,
    {
        DistWithState::new(self)
    }
}
