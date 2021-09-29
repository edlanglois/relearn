//! Reinforcement learning environments
mod bandits;
mod builders;
mod chain;
mod mdps;
mod memory;
mod meta;
mod stateful;
#[cfg(test)]
pub mod testing;
mod wrappers;

pub use bandits::{
    Bandit, BernoulliBandit, DeterministicBandit, OneHotBandits, UniformBernoulliBandits,
};
pub use builders::{BuildEnv, BuildEnvDist, BuildEnvError, BuildPomdp, BuildPomdpDist, CloneBuild};
pub use chain::Chain;
pub use mdps::DirichletRandomMdps;
pub use memory::MemoryGame;
pub use meta::{
    InnerEnvStructure, MetaEnv, MetaEnvConfig, MetaObservationSpace, MetaPomdp, MetaState,
};
pub use stateful::{IntoEnv, PomdpEnv};
pub use wrappers::{StepLimit, WithStepLimit, Wrapped};

use crate::spaces::Space;
use rand::{rngs::StdRng, Rng};
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
pub struct StoredEnvStructure<OS, AS> {
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

/// A Markov decision process (MDP).
///
/// The concept of an episode is an abstraction on the MDP formalism.
/// An episode ending means that all possible future trajectories have 0 reward on each step.
pub trait Mdp {
    type State;
    type Action;

    /// Sample a new initial state.
    fn initial_state(&self, rng: &mut StdRng) -> Self::State;

    /// Sample a state transition.
    ///
    /// # Returns
    /// * `state`: The resulting state.
    ///     Is `None` if the resulting state is terminal.
    ///     All trajectories from terminal states yield 0 reward on each step.
    /// * `reward`: The reward value for this transition.
    /// * `episode_done`: Whether this step ends the current episode.
    ///     - If `observation` is `None` then `episode_done` must be true.
    ///     - An episode may be done for other reasons, like a step limit.
    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool);
}

/// A partially observable Markov decision process (POMDP).
///
/// The concept of an episode is an abstraction on the MDP formalism.
/// An episode ending means that all possible future trajectories have 0 reward on each step.
pub trait Pomdp {
    type State;
    type Observation;
    type Action;

    /// Sample a new initial state.
    fn initial_state(&self, rng: &mut StdRng) -> Self::State;

    /// Sample an observation for a state.
    fn observe(&self, state: &Self::State, rng: &mut StdRng) -> Self::Observation;

    /// Sample a state transition.
    ///
    /// # Returns
    /// * `state`: The resulting state.
    ///     Is `None` if the resulting state is terminal.
    ///     All trajectories from terminal states yield 0 reward on each step.
    /// * `reward`: The reward value for this transition.
    /// * `episode_done`: Whether this step ends the episode.
    ///     - If `observation` is `None` then `episode_done` must be true.
    ///     - An episode may be done for other reasons, like a step limit.
    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool);
}

impl<E: Mdp> Pomdp for E
where
    <Self as Mdp>::State: Copy,
{
    type State = <Self as Mdp>::State;
    type Observation = <Self as Mdp>::State;
    type Action = <Self as Mdp>::Action;

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        Mdp::initial_state(self, rng)
    }

    fn observe(&self, state: &Self::State, _: &mut StdRng) -> Self::Observation {
        *state
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        Mdp::step(self, state, action, rng)
    }
}

/// A reinforcement learning environment with internal state.
///
/// Prefer implementing [`Pomdp`] since [`PomdpEnv`] can be used
/// to create an [`Environment`] out of a [`Pomdp`].
pub trait Environment {
    type Observation;
    type Action;

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
    fn step(&mut self, action: &Self::Action) -> (Option<Self::Observation>, f64, bool);

    /// Reset the environment to an initial state.
    ///
    /// Must be called before each new episode.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    fn reset(&mut self) -> Self::Observation;
}

impl<E: Environment + ?Sized> Environment for Box<E> {
    type Observation = E::Observation;
    type Action = E::Action;

    fn step(&mut self, action: &Self::Action) -> (Option<Self::Observation>, f64, bool) {
        E::step(self, action)
    }

    fn reset(&mut self) -> Self::Observation {
        E::reset(self)
    }
}

/// A distribution of [`Pomdp`] sharing the same external structure.
///
/// The [`EnvStructure`] of each sampled environment must be a subset of the `EnvStructure` of the
/// distribution as a whole. The discount factors must be identical.
/// The transition dynamics of the individual environment samples may differ.
pub trait PomdpDistribution: EnvStructure {
    type Pomdp: Pomdp<
            Observation = <<Self as EnvStructure>::ObservationSpace as Space>::Element,
            Action = <<Self as EnvStructure>::ActionSpace as Space>::Element,
        > + EnvStructure<
            ObservationSpace = <Self as EnvStructure>::ObservationSpace,
            ActionSpace = <Self as EnvStructure>::ActionSpace,
        >;

    /// Sample a POMDP from the distribution.
    ///
    /// # Args
    /// * `rng` - Random number generator used for sampling the environment structure.
    fn sample_pomdp(&self, rng: &mut StdRng) -> Self::Pomdp;
}

/// A distribution of environments sharing the same external structure.
///
/// The [`EnvStructure`] of each sampled environment must be a subset of the `EnvStructure` of the
/// distribution as a whole. The discount factors must be identical.
/// The transition dynamics of the individual environment samples may differ.
pub trait EnvDistribution: EnvStructure {
    type Environment: EnvStructure<
        ObservationSpace = <Self as EnvStructure>::ObservationSpace,
        ActionSpace = <Self as EnvStructure>::ActionSpace,
    >;

    /// Sample an environment from the distribution.
    ///
    /// # Args
    /// * `rng` - Random number generator used for sampling the environment structure and for
    ///           seeding any internal randomness of the environment dynamics.
    fn sample_environment(&self, rng: &mut StdRng) -> Self::Environment;
}

impl<T> EnvDistribution for T
where
    T: PomdpDistribution,
    <T as PomdpDistribution>::Pomdp: Pomdp,
{
    type Environment = PomdpEnv<T::Pomdp>;

    fn sample_environment(&self, rng: &mut StdRng) -> Self::Environment {
        PomdpEnv::new(self.sample_pomdp(rng), rng.gen())
    }
}
