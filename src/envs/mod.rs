//! Reinforcement learning environments
mod bandits;
mod builders;
mod cartpole;
mod chain;
mod mdps;
mod memory;
mod meta;
mod multiagent;
mod partition;
#[cfg(test)]
pub mod testing;
mod wrappers;

pub use bandits::{
    Bandit, BernoulliBandit, DeterministicBandit, OneHotBandits, UniformBernoulliBandits,
};
pub use builders::{BuildEnv, BuildEnvDist, BuildEnvError, CloneBuild};
pub use cartpole::{CartPole, CartPoleConfig};
pub use chain::Chain;
pub use mdps::DirichletRandomMdps;
pub use memory::MemoryGame;
pub use meta::{InnerEnvStructure, MetaEnv, MetaObservation, MetaObservationSpace, MetaState};
pub use multiagent::fruit::{self, FruitGame};
pub use multiagent::views::{FirstPlayerView, SecondPlayerView};
pub use partition::PartitionGame;
pub use wrappers::{StepLimit, WithStepLimit, Wrapped};

use crate::agents::Actor;
use crate::logging::StatsLogger;
use crate::simulation::{SimSeed, SimulatorSteps};
use crate::spaces::Space;
use crate::Prng;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::f64;

/// A reinforcement learning [`Environment`] with consistent [`EnvStructure`].
///
/// # Design Discussion
/// [`EnvStructure`] is not a supertrait of [`Environment`] because knowing the observation and
/// action spaces is not necessary for simulation, only the observation and action types must be
/// known.
///
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
        > + ?Sized
{
}

/// A reinforcement learning environment.
///
/// Formally, this is a Partially Observable Markov Decision Process (POMDP) with episodes.
/// An episode is a sequence of environment steps starting with [`Environment::initial_state`]
/// and ending when [`Environment::step`] returns either
/// * [`Successor::Terminate`] meaning all possible future rewards are zero; or
/// * [`Successor::Interrupt`] meaning the POMDP would continue with possible nonzero reward but
///     but has been prematurely interrupted.
///
/// This trait encodes the dynamics of a reinforcement learning environment.
/// The actual state is represented by the `State` associated type.
///
/// # Design Discussion
/// ## `State`
/// The use of an explicit `State` associated type allows the type system to manage episode
/// lifetimes; there is no possibility of an incomplete reset between episodes.
/// However, it forces the users of this trait to handle `State` when they might prefer it to be
/// a hidden internal implementation detail.
/// Once [Generic Associated Types][GAT] are stable, an alternative [`Environment`] trait could
/// have an `Episode<'a>` associated type where `Episode` provides a `step` method and
/// internally manages state.
/// However, using the generic `Episode<'a>` approach would make it difficult to store an
/// environment and an episode together.
/// Something similar could be done without GAT using an
/// `Episode<'a, E: Environment>(&'a E, E::State)` struct with the same drawbacks.
///
/// ## Random State
/// The episode is not responsible for managing its own pseudo-random state.
/// This avoids having to frequently re-initialize the random number generator on each episode and
/// simplifies  state definitions.
///
/// [GAT]: https://rust-lang.github.io/rfcs/1598-generic_associated_types.html
pub trait Environment {
    type State;
    type Observation;
    type Action;

    /// Sample a state for the start of a new episode.
    ///
    /// `rng` is a source of randomness for sampling the initial state.
    /// This includes seeding any pseudo-random number generators used by the environment, which
    /// must be stored within `State`.
    fn initial_state(&self, rng: &mut Prng) -> Self::State;

    /// Generate an observation for a given state.
    fn observe(&self, state: &Self::State, rng: &mut Prng) -> Self::Observation;

    /// Perform a state transition in reponse to an action.
    ///
    /// # Args
    /// * `state`  - The initial state.
    /// * `action` - The action to take at this state.
    /// * `logger` - Logger for any auxiliary information.
    ///
    /// # Returns
    /// * `successor` - The resulting state or outcome.
    /// * `reward` - The reward value for this transition.
    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, f64);

    /// Run this environment with the given actor.
    fn run<T, L>(self, actor: T, seed: SimSeed, logger: L) -> SimulatorSteps<Self, T, Prng, L>
    where
        T: Actor<Self::Observation, Self::Action>,
        L: StatsLogger,
        Self: Sized,
    {
        SimulatorSteps::new_seeded(self, actor, seed, logger)
    }

    /// Wrap the environment in an episode step limit.
    fn with_step_limit(self, max_steps_per_episode: u64) -> WithStepLimit<Self>
    where
        Self: Sized,
    {
        Wrapped::new(self, StepLimit::new(max_steps_per_episode))
    }
}

/// Implement `Environment` for a deref-able wrapper type generic over `T: Environment + ?Sized`.
macro_rules! impl_wrapped_environment {
    ($wrapper:ty) => {
        impl<T: Environment + ?Sized> Environment for $wrapper {
            type State = T::State;
            type Observation = T::Observation;
            type Action = T::Action;
            fn initial_state(&self, rng: &mut Prng) -> Self::State {
                T::initial_state(self, rng)
            }
            fn observe(&self, state: &Self::State, rng: &mut Prng) -> Self::Observation {
                T::observe(self, state, rng)
            }
            fn step(
                &self,
                state: Self::State,
                action: &Self::Action,
                rng: &mut Prng,
                logger: &mut dyn StatsLogger,
            ) -> (Successor<Self::State>, f64) {
                T::step(self, state, action, rng, logger)
            }
        }
    };
}
impl_wrapped_environment!(&'_ T);
impl_wrapped_environment!(Box<T>);

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
    /// Every element in this space must be a valid action in all environment states (although
    /// immediately ending the episode with negative reward is a possible outcome).
    /// The environment may misbehave or panic for actions outside of this action space.
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

/// Implement `EnvStructure` for a deref-able wrapper type generic over `T: EnvStructure + ?Sized`.
macro_rules! impl_wrapped_env_structure {
    ($wrapper:ty) => {
        impl<T: EnvStructure + ?Sized> EnvStructure for $wrapper {
            type ObservationSpace = T::ObservationSpace;
            type ActionSpace = T::ActionSpace;

            fn observation_space(&self) -> Self::ObservationSpace {
                T::observation_space(self)
            }
            fn action_space(&self) -> Self::ActionSpace {
                T::action_space(self)
            }
            fn reward_range(&self) -> (f64, f64) {
                T::reward_range(self)
            }
            fn discount_factor(&self) -> f64 {
                T::discount_factor(self)
            }
        }
    };
}
impl_wrapped_env_structure!(&'_ T);
impl_wrapped_env_structure!(Box<T>);

/// The successor state or outcome of an episode step.
///
/// The purpose of the second generic parameter `U` is to control the ownership of the following
/// state or observation when the episode continues. By default the successor is owned but it can
/// also be borrowed `U = &T` or omitted `U = ()`. This is useful because users might want to
/// extract the next observation without copying.
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

    /// Partition into a `PartialSuccessor` and the `Continue` state, if any.
    #[allow(clippy::missing_const_for_fn)] // not allowed to be const at time of writing
    #[inline]
    pub fn into_partial_continue(self) -> (PartialSuccessor<T>, Option<U>) {
        match self {
            Self::Continue(o) => (Successor::Continue(()), Some(o)),
            Self::Terminate => (Successor::Terminate, None),
            Self::Interrupt(o) => (Successor::Interrupt(o), None),
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

/// A distribution of [`Environment`] sharing the same external structure.
///
/// The [`EnvStructure`] of each sampled environment must be a subset of the `EnvStructure` of the
/// distribution as a whole. The discount factors must be identical.
/// The transition dynamics of the individual environment samples may differ.
pub trait EnvDistribution: EnvStructure {
    type Environment: StructuredEnvironment<
        ObservationSpace = Self::ObservationSpace,
        ActionSpace = Self::ActionSpace,
    >;

    /// Sample an environment from the distribution.
    ///
    /// # Args
    /// * `rng` - Random number generator used for sampling the environment structure.
    fn sample_environment(&self, rng: &mut Prng) -> Self::Environment;
}
