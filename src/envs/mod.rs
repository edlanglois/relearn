//! Reinforcement learning environments
mod bandits;
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
pub use chain::Chain;
pub use mdps::DirichletRandomMdps;
pub use memory::MemoryGame;
pub use meta::{
    InnerEnvStructure, MetaEnv, MetaEnvConfig, MetaEnvState, MetaObservationSpace, StatefulMetaEnv,
};
pub use stateful::{IntoStateful, PomdpEnv, WithState};
pub use step_limit::StepLimit;

use crate::spaces::Space;
use rand::distributions::BernoulliError;
use rand::{rngs::StdRng, Rng};
use std::convert::Infallible;
use std::f64;
use thiserror::Error;

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
/// to create an `Environment` out of a `Pomdp`.
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

/// Builds an environment
pub trait BuildEnv<E> {
    /// Build an environment instance.
    ///
    /// # Args
    /// * `seed` - Seed for pseudo-randomness used by the environment.
    ///     Includes both randomization of the environment structure, and
    ///     random sampling of step outcomes within this structure.
    fn build_env(&self, seed: u64) -> Result<E, BuildEnvError>;
}

/// Error building an environment
#[derive(Debug, Clone, PartialEq, Error)]
pub enum BuildEnvError {
    #[error(transparent)]
    BernoulliError(#[from] BernoulliError),
}

impl From<Infallible> for BuildEnvError {
    fn from(_: Infallible) -> Self {
        unreachable!();
    }
}

/// A distribution of environments sharing the same structure.
///
/// The spaces / intervals of each sampled environment must be equal to
/// or a subset of the spaces for `EnvDistribution`.
/// The discount factor of the sampled environments must be the same.
pub trait EnvDistribution {
    type Environment;

    /// Sample an environment from the distribution.
    ///
    /// # Args
    /// * `rng` - Random number generator used for sampling the environment structure and for
    ///           seeding any internal randomness of the environment dynamics.
    fn sample_environment(&self, rng: &mut StdRng) -> Self::Environment;

    /// Apply a wrapper to sampled environments.
    fn wrap<W>(self, wrapper: W) -> Wrapped<Self, W>
    where
        Self: Sized,
    {
        Wrapped {
            inner: self,
            wrapper,
        }
    }
}

/// Builds an environment distribution.
pub trait BuildEnvDist<E> {
    fn build_env_dist(&self) -> E;
}

/// Can wrap an environment.
pub trait EnvWrapper<E> {
    type Wrapped;
    /// Wrap an environment
    ///
    /// # Args
    /// * `env` - Environment to wrap.
    /// * `rng` - Pseudorandom number generator for setting any random state.
    fn wrap<R: Rng + ?Sized>(&self, env: E, rng: &mut R) -> Self::Wrapped;
}

/// Transforms an [`EnvStructure`] for a wrapper.
///
/// Should be consistent with `<Self as [EnvWrapper]>::Wrapped as [EnvStructure]` if applicable.
pub trait EnvStructureWrapper<E> {
    type WrappedObservationSpace: Space;
    type WrappedActionSpace: Space;

    /// Wrapped observation space
    fn observation_space(&self, env: &E) -> Self::WrappedObservationSpace;
    /// Wrapped action space
    fn action_space(&self, env: &E) -> Self::WrappedActionSpace;
    /// Wrapped reward range
    fn reward_range(&self, env: &E) -> (f64, f64);
    /// Wrapped discount factor
    fn discount_factor(&self, env: &E) -> f64;
}

// TODO: Allow implementing EnvStructureWrapper without conflict

/// A wrapper with the same [`EnvStructure`] space types as the inner wrapped value.
///
/// By default the structure methods forward the inner values but these can be overwritten.
pub trait InnerStructureWrapper<E: EnvStructure> {
    fn observation_space(&self, env: &E) -> E::ObservationSpace {
        env.observation_space()
    }

    fn action_space(&self, env: &E) -> E::ActionSpace {
        env.action_space()
    }

    fn reward_range(&self, env: &E) -> (f64, f64) {
        env.reward_range()
    }

    fn discount_factor(&self, env: &E) -> f64 {
        env.discount_factor()
    }
}

impl<W, E> EnvStructureWrapper<E> for W
where
    W: InnerStructureWrapper<E>,
    E: EnvStructure,
{
    type WrappedObservationSpace = E::ObservationSpace;
    type WrappedActionSpace = E::ActionSpace;

    fn observation_space(&self, env: &E) -> E::ObservationSpace {
        InnerStructureWrapper::observation_space(self, env)
    }

    fn action_space(&self, env: &E) -> E::ActionSpace {
        InnerStructureWrapper::action_space(self, env)
    }

    fn reward_range(&self, env: &E) -> (f64, f64) {
        InnerStructureWrapper::reward_range(self, env)
    }

    fn discount_factor(&self, env: &E) -> f64 {
        InnerStructureWrapper::discount_factor(self, env)
    }
}

/// A basic wrapped object.
///
/// Consists of the inner object and the wrapper state.
///
/// May be used as an environment wrapper as `Wrapped<T, SomeWrapper>`
/// where `SomeWrapper: EnvWrapper + EnvStructureWrapper`
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wrapped<T, W> {
    /// Wrapped object
    pub inner: T,
    /// The wrapper
    pub wrapper: W,
}

impl<T, W> Wrapped<T, W> {
    pub const fn new(inner: T, wrapper: W) -> Self {
        Self { inner, wrapper }
    }
}

// TODO: Maybe provide a macro for these since they conflict with implementing builders for B
/*
/// A wrapped BuildEnv builds wrapped environments.
impl<B, E, W> BuildEnv<Wrapped<E, W>> for Wrapped<B, W>
where
    B: BuildEnv<E>,
    W: Clone,
{
    fn build_env(&self, seed: u64) -> Result<Wrapped<E, W>, BuildEnvError> {
        Ok(Wrapped {
            inner: self.inner.build_env(seed)?,
            wrapper: self.wrapper.clone(),
        })
    }
}

/// A wrapped BuildEnvDist builds wrapped distributions.
impl<B, T, W> BuildEnvDist<Wrapped<T, W>> for Wrapped<B, W>
where
    B: BuildEnvDist<T>,
    W: Clone,
{
    fn build_env_dist(&self) -> Wrapped<T, W> {
        Wrapped {
            inner: self.inner.build_env_dist(),
            wrapper: self.wrapper.clone(),
        }
    }
}
*/

impl<T, W> EnvStructure for Wrapped<T, W>
where
    W: EnvStructureWrapper<T>,
{
    type ObservationSpace = W::WrappedObservationSpace;
    type ActionSpace = W::WrappedActionSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.wrapper.observation_space(&self.inner)
    }
    fn action_space(&self) -> Self::ActionSpace {
        self.wrapper.action_space(&self.inner)
    }
    fn reward_range(&self) -> (f64, f64) {
        self.wrapper.reward_range(&self.inner)
    }
    fn discount_factor(&self) -> f64 {
        self.wrapper.discount_factor(&self.inner)
    }
}

impl<T, W> EnvDistribution for Wrapped<T, W>
where
    T: EnvDistribution,
    // W can wrap both the environment distribution T and the inner environment T::Environment.
    // The wrapped environment structures are consistent with each other.
    W: EnvStructureWrapper<T> + EnvWrapper<T::Environment>,
    <W as EnvWrapper<T::Environment>>::Wrapped: EnvStructure<
        ObservationSpace = <W as EnvStructureWrapper<T>>::WrappedObservationSpace,
        ActionSpace = <W as EnvStructureWrapper<T>>::WrappedActionSpace,
    >,
{
    type Environment = <W as EnvWrapper<T::Environment>>::Wrapped;

    fn sample_environment(&self, rng: &mut StdRng) -> Self::Environment {
        self.wrapper.wrap(self.inner.sample_environment(rng), rng)
    }
}
