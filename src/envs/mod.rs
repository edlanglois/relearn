//! Reinforcement learning environments
mod bandits;
mod builders;
mod cartpole;
mod chain;
mod mdps;
mod memory;
mod meta;
mod multiagent;
mod stateful;
#[cfg(test)]
pub mod testing;
mod wrappers;

pub use bandits::{
    Bandit, BernoulliBandit, DeterministicBandit, OneHotBandits, UniformBernoulliBandits,
};
pub use builders::{BuildEnv, BuildEnvDist, BuildEnvError, BuildPomdp, BuildPomdpDist, CloneBuild};
pub use cartpole::{CartPole, CartPoleConfig};
pub use chain::Chain;
pub use mdps::DirichletRandomMdps;
pub use memory::MemoryGame;
pub use meta::{
    InnerEnvStructure, MetaEnv, MetaEnvConfig, MetaObservation, MetaObservationSpace, MetaPomdp,
    MetaState,
};
pub use multiagent::fruit::FruitGame;
pub use multiagent::views::{FirstPlayerView, SecondPlayerView};
pub use stateful::{IntoEnv, PomdpEnv};
pub use wrappers::{StepLimit, WithStepLimit, Wrapped};

use crate::agents::Actor;
use crate::logging::{Logger, TimeSeriesLogger};
use crate::simulation::SimSteps;
use crate::spaces::Space;
use rand::{rngs::StdRng, Rng};
use std::borrow::Borrow;
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

impl<E: EnvStructure + ?Sized> EnvStructure for &'_ E {
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

/// The successor state or outcome of an episode step.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Successor<T, U = T> {
    /// The episode continues with the given state.
    Continue(U),
    /// The episode ends by entering a terminal state.
    ///
    /// A terminal state is one from which all possible trajectories would have 0 reward.
    Terminate,
    /// The episode ends despite entering the given non-terminal state.
    ///
    /// Had the episode continued, non-zero future rewards might have been possible.
    /// For example, the episode may have been interrupted due to a step limit.
    Interrupt(T),
}
impl<T, U> Successor<T, U> {
    /// Get the inner state of `Successor::Continue`
    #[allow(clippy::missing_const_for_fn)] // not allowed to be const at time of writing
    #[inline]
    pub fn continue_(self) -> Option<U> {
        match self {
            Self::Continue(s) => Some(s),
            _ => None,
        }
    }

    /// Get the inner state of `Successor::Interrupt`
    #[allow(clippy::missing_const_for_fn)] // not allowed to be const at time of writing
    #[inline]
    pub fn interrupt(self) -> Option<T> {
        match self {
            Self::Interrupt(s) => Some(s),
            _ => None,
        }
    }

    /// Whether this successor marks the end of an episode
    #[inline]
    pub const fn episode_done(&self) -> bool {
        !matches!(self, Successor::Continue(_))
    }

    /// Drop any stored `Continue` state, converting into `PartialSuccessor`.
    #[allow(clippy::missing_const_for_fn)] // not allowed to be const at time of writing
    #[inline]
    pub fn into_partial(self) -> PartialSuccessor<T> {
        match self {
            Self::Continue(_) => Successor::Continue(()),
            Self::Terminate => Successor::Terminate,
            Self::Interrupt(s) => Successor::Interrupt(s),
        }
    }

    /// Apply a function to just the successor `Continue` variant.
    #[inline]
    pub fn map_continue<F, V>(self, f: F) -> Successor<T, V>
    where
        F: FnOnce(U) -> V,
    {
        match self {
            Self::Continue(s) => Successor::Continue(f(s)),
            Self::Terminate => Successor::Terminate,
            Self::Interrupt(s) => Successor::Interrupt(s),
        }
    }

    /// Take the `Continue` value or compute it from a closure.
    ///
    /// Also returns the rest of the `Successor` as a `PartialSuccessor`.
    #[inline]
    pub fn take_continue_or_else<F>(self, f: F) -> (PartialSuccessor<T>, U)
    where
        F: FnOnce() -> U,
    {
        match self {
            Self::Continue(o) => (Successor::Continue(()), o),
            Self::Terminate => (Successor::Terminate, f()),
            Self::Interrupt(o) => (Successor::Interrupt(o), f()),
        }
    }
}

impl<T> Successor<T> {
    /// Apply a transformation to the inner state when present.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Successor<U> {
        match self {
            Self::Continue(state) => Successor::Continue(f(state)),
            Self::Terminate => Successor::Terminate,
            Self::Interrupt(state) => Successor::Interrupt(f(state)),
        }
    }

    /// Get the inner state of `Continue` and `Interrupt` variants.
    #[allow(clippy::missing_const_for_fn)] // not allowed to be const at time of writing
    #[inline]
    pub fn into_inner(self) -> Option<T> {
        match self {
            Self::Continue(s) => Some(s),
            Self::Interrupt(s) => Some(s),
            Self::Terminate => None,
        }
    }
}

impl<T, U: Borrow<T>> Successor<T, U> {
    /// Convert `&Successor<T, U>` to `Successor<&T>`.
    #[inline]
    pub fn as_ref(&self) -> Successor<&T> {
        match self {
            Self::Continue(s) => Successor::Continue(s.borrow()),
            Self::Terminate => Successor::Terminate,
            Self::Interrupt(s) => Successor::Interrupt(s),
        }
    }
}

impl<T: Clone, U: Clone> Successor<&'_ T, &'_ U> {
    /// Convert `Successor<&T, &U>` to `Successor<T, U>` by cloning its contents
    #[inline]
    pub fn cloned(self) -> Successor<T, U> {
        match self {
            Self::Continue(s) => Successor::Continue(s.clone()),
            Self::Terminate => Successor::Terminate,
            Self::Interrupt(s) => Successor::Interrupt(s.clone()),
        }
    }
}

impl<T: Clone> Successor<T, &'_ T> {
    /// Convert into an owned successor by cloning any borrowed successor observation.
    #[inline]
    pub fn into_owned(self) -> Successor<T> {
        match self {
            Self::Continue(s) => Successor::Continue(s.clone()),
            Self::Terminate => Successor::Terminate,
            Self::Interrupt(s) => Successor::Interrupt(s),
        }
    }
}

/// A successor that only stores a reference to the successor state if continuing.
pub type RefSuccessor<'a, T> = Successor<T, &'a T>;

/// A successor that does not store the successor state if continuing.
pub type PartialSuccessor<T> = Successor<T, ()>;

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

impl<OS, AS> StoredEnvStructure<OS, AS> {
    pub const fn new(
        observation_space: OS,
        action_space: AS,
        reward_range: (f64, f64),
        discount_factor: f64,
    ) -> Self {
        Self {
            observation_space,
            action_space,
            reward_range,
            discount_factor,
        }
    }
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
    /// # Args
    /// * `state`  - The initial state.
    /// * `action` - The action to take at this state.
    /// * `rng`    - Random number generator for any stochasticity in the transition.
    /// * `logger` - Logger for any auxiliary information.
    ///
    /// # Returns
    /// * `successor` - The resulting state or outcome.
    /// * `reward` - The reward value for this transition.
    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
        logger: &mut dyn Logger,
    ) -> (Successor<Self::State>, f64);
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
    /// # Args
    /// * `state`  - The initial state.
    /// * `action` - The action to take at this state.
    /// * `rng`    - Random number generator for any stochasticity in the transition.
    /// * `logger` - Logger for any auxiliary information.
    ///
    /// # Returns
    /// * `successor` - The resulting state or outcome.
    /// * `reward` - The reward value for this transition.
    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
        logger: &mut dyn Logger,
    ) -> (Successor<Self::State>, f64);
}

impl<E: Mdp> Pomdp for E
where
    E::State: Copy,
{
    type State = E::State;
    type Observation = E::State;
    type Action = E::Action;

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
        logger: &mut dyn Logger,
    ) -> (Successor<Self::State>, f64) {
        Mdp::step(self, state, action, rng, logger)
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
    /// * `successor` - The resulting observation or outcome.
    /// * `reward` - The reward value for this transition.
    fn step(
        &mut self,
        action: &Self::Action,
        logger: &mut dyn Logger,
    ) -> (Successor<Self::Observation>, f64);

    /// Reset the environment to an initial state.
    ///
    /// Must be called before each new episode.
    ///
    /// # Returns
    /// * `observation`: An observation of the resulting state.
    fn reset(&mut self) -> Self::Observation;

    /// Run this environment with the given actor.
    fn run<A, L>(self, actor: A, logger: L) -> SimSteps<Self, A, L>
    where
        A: Actor<Self::Observation, Self::Action>,
        L: TimeSeriesLogger,
        Self: Sized,
    {
        SimSteps::new(self, actor, logger)
    }
}

impl<E: Environment + ?Sized> Environment for &'_ mut E {
    type Observation = E::Observation;
    type Action = E::Action;

    fn step(
        &mut self,
        action: &Self::Action,
        logger: &mut dyn Logger,
    ) -> (Successor<Self::Observation>, f64) {
        E::step(self, action, logger)
    }

    fn reset(&mut self) -> Self::Observation {
        E::reset(self)
    }
}

impl<E: Environment + ?Sized> Environment for Box<E> {
    type Observation = E::Observation;
    type Action = E::Action;

    fn step(
        &mut self,
        action: &Self::Action,
        logger: &mut dyn Logger,
    ) -> (Successor<Self::Observation>, f64) {
        E::step(self, action, logger)
    }

    fn reset(&mut self) -> Self::Observation {
        E::reset(self)
    }
}

/// An [`Environment`] with consistent [`EnvStructure`].
///
/// The reason that [`EnvStructure`] is not already a supertrait of [`Environment`] is because when
/// simulating environments we only need to know `Observation` and `Action`, not `ObservationSpace`
/// and `ActionSpace`. The trait object `dyn Environment<Observation=_, Action=_>` can be used
/// without also having to monomorphize over the space.
pub trait StructuredEnvironment:
    EnvStructure
    + Environment<
        Observation = <Self::ObservationSpace as Space>::Element,
        Action = <Self::ActionSpace as Space>::Element,
    >
{
}
impl<T> StructuredEnvironment for T where
    T: EnvStructure
        + Environment<
            Observation = <Self::ObservationSpace as Space>::Element,
            Action = <Self::ActionSpace as Space>::Element,
        >
{
}

/// A distribution of [`Pomdp`] sharing the same external structure.
///
/// The [`EnvStructure`] of each sampled environment must be a subset of the `EnvStructure` of the
/// distribution as a whole. The discount factors must be identical.
/// The transition dynamics of the individual environment samples may differ.
pub trait PomdpDistribution: EnvStructure {
    type Pomdp: Pomdp<
            Observation = <Self::ObservationSpace as Space>::Element,
            Action = <Self::ActionSpace as Space>::Element,
        > + EnvStructure<ObservationSpace = Self::ObservationSpace, ActionSpace = Self::ActionSpace>;

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
    type Environment: Environment<
            Observation = <Self::ObservationSpace as Space>::Element,
            Action = <Self::ActionSpace as Space>::Element,
        > + EnvStructure<ObservationSpace = Self::ObservationSpace, ActionSpace = Self::ActionSpace>;

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
{
    type Environment = PomdpEnv<T::Pomdp>;

    fn sample_environment(&self, rng: &mut StdRng) -> Self::Environment {
        PomdpEnv::new(self.sample_pomdp(rng), rng.gen())
    }
}
