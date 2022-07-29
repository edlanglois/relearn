use super::super::{EnvStructure, Environment, Successor};
use super::{StructurePreservingWrapper, Wrapped};
use crate::logging::StatsLogger;
use crate::spaces::IntervalSpace;
use crate::Prng;
use serde::{Deserialize, Serialize};

/// Environment wrapper that interrupts episodes after a set number of steps.
///
/// The step limit is not included in the observation space.
/// This can make it hard or impossible for the agent to accurately model the return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LatentStepLimit {
    /// Maximum number of steps per episode
    pub max_steps_per_episode: u64,
}

impl LatentStepLimit {
    #[must_use]
    #[inline]
    pub const fn new(max_steps_per_episode: u64) -> Self {
        assert!(max_steps_per_episode > 0, "step limit must be positive");
        Self {
            max_steps_per_episode,
        }
    }
}

impl Default for LatentStepLimit {
    #[inline]
    fn default() -> Self {
        Self {
            max_steps_per_episode: 100,
        }
    }
}

/// Wrap an environment with a per-episode step limit.
pub type WithLatentStepLimit<E> = Wrapped<E, LatentStepLimit>;

impl StructurePreservingWrapper for LatentStepLimit {}

/// Wrapped environment state with a step limit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepLimitState<T> {
    pub inner: T,
    pub steps_remaining: u64,
}

impl<E: Environment> Environment for Wrapped<E, LatentStepLimit> {
    /// `(inner_state, step_count)`
    type State = StepLimitState<E::State>;
    type Observation = E::Observation;
    type Action = E::Action;
    type Feedback = E::Feedback;

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        StepLimitState {
            inner: self.inner.initial_state(rng),
            steps_remaining: self.wrapper.max_steps_per_episode,
        }
    }

    fn observe(&self, state: &Self::State, rng: &mut Prng) -> Self::Observation {
        self.inner.observe(&state.inner, rng)
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        assert!(
            state.steps_remaining > 0,
            "invalid step from a state with no remaining steps"
        );
        let (inner_successor, feedback) = self.inner.step(state.inner, action, rng, logger);

        // Decrement remaining steps and interrupt if none remain
        let successor = inner_successor
            .map(|inner| StepLimitState {
                inner,
                steps_remaining: state.steps_remaining - 1,
            })
            .then_interrupt_if(|next_state| next_state.steps_remaining == 0);
        (successor, feedback)
    }
}

/// Environment wrapper that interrupts episodes after a set number of steps.
///
/// The amount of elapsed steps out of the limit is included in the observation as a floating-point
/// number between 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisibleStepLimit {
    /// Maximum number of steps per episode
    pub max_steps_per_episode: u64,
}

impl VisibleStepLimit {
    #[must_use]
    #[inline]
    pub const fn new(max_steps_per_episode: u64) -> Self {
        assert!(max_steps_per_episode > 0, "step limit must be positive");
        Self {
            max_steps_per_episode,
        }
    }
}

impl Default for VisibleStepLimit {
    #[inline]
    fn default() -> Self {
        Self {
            max_steps_per_episode: 100,
        }
    }
}

/// Wrap an environment with a per-episode step limit.
pub type WithVisibleStepLimit<E> = Wrapped<E, VisibleStepLimit>;

/// Wrapped environment observation with a step limit.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepLimitObs<T> {
    pub inner: T,
    /// Fraction of the episode remaining.
    pub remaining: f64,
}

#[derive(Debug, Copy, Clone, PartialEq, ProductSpace, Serialize, Deserialize)]
#[element(StepLimitObs<T::Element>)]
pub struct StepLimitObsSpace<T> {
    pub inner: T,
    pub remaining: IntervalSpace<f64>,
}

impl<T> From<T> for StepLimitObsSpace<T> {
    fn from(inner: T) -> Self {
        StepLimitObsSpace {
            inner,
            remaining: IntervalSpace::new(0.0, 1.0),
        }
    }
}

impl<T: Default> Default for StepLimitObsSpace<T> {
    fn default() -> Self {
        StepLimitObsSpace {
            inner: T::default(),
            remaining: IntervalSpace::new(0.0, 1.0),
        }
    }
}

impl<T: EnvStructure> EnvStructure for Wrapped<T, VisibleStepLimit> {
    type ObservationSpace = StepLimitObsSpace<T::ObservationSpace>;
    type ActionSpace = T::ActionSpace;
    type FeedbackSpace = T::FeedbackSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.inner.observation_space().into()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space()
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        self.inner.feedback_space()
    }

    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor()
    }
}

impl<E: Environment> Environment for Wrapped<E, VisibleStepLimit> {
    /// `(inner_state, step_count)`
    type State = StepLimitState<E::State>;
    type Observation = StepLimitObs<E::Observation>;
    type Action = E::Action;
    type Feedback = E::Feedback;

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        StepLimitState {
            inner: self.inner.initial_state(rng),
            steps_remaining: self.wrapper.max_steps_per_episode,
        }
    }

    fn observe(&self, state: &Self::State, rng: &mut Prng) -> Self::Observation {
        let remaining = state.steps_remaining as f64 / self.wrapper.max_steps_per_episode as f64;
        StepLimitObs {
            inner: self.inner.observe(&state.inner, rng),
            remaining,
        }
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        assert!(
            state.steps_remaining > 0,
            "invalid step from a state with no remaining steps"
        );
        let (inner_successor, feedback) = self.inner.step(state.inner, action, rng, logger);

        // Decrement remaining steps and interrupt if none remain
        let successor = inner_successor
            .map(|inner| StepLimitState {
                inner,
                steps_remaining: state.steps_remaining - 1,
            })
            .then_interrupt_if(|next_state| next_state.steps_remaining == 0);
        (successor, feedback)
    }
}

#[cfg(test)]
mod latent {
    use super::super::super::{chain::Move, testing, BuildEnv, Chain};
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_default() {
        testing::check_structured_env(&WithLatentStepLimit::<Chain>::default(), 1000, 119);
    }

    #[test]
    fn build() {
        let config = WithLatentStepLimit::<Chain>::default();
        let _env = config.build_env(&mut Prng::seed_from_u64(0)).unwrap();
    }

    #[test]
    fn step_limit() {
        let mut rng = Prng::seed_from_u64(110);
        let env = WithLatentStepLimit::new(Chain::default(), LatentStepLimit::new(2));
        let state = env.initial_state(&mut rng);

        // Step 1
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Continue(_)));
        let state = successor.into_continue().unwrap();

        // Step 2
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Interrupt(_)));
    }
}

#[cfg(test)]
mod visible {
    use super::super::super::{chain::Move, testing, BuildEnv, Chain};
    use super::super::Wrap;
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_default() {
        testing::check_structured_env(
            &Chain::default().wrap(VisibleStepLimit::default()),
            1000,
            119,
        );
    }

    #[test]
    fn build() {
        let config = Chain::default().wrap(VisibleStepLimit::default());
        let _env = config.build_env(&mut Prng::seed_from_u64(0)).unwrap();
    }

    #[test]
    #[allow(clippy::float_cmp)] // expecting exact values
    fn step_limit() {
        let mut rng = Prng::seed_from_u64(110);
        let env = WithVisibleStepLimit::new(Chain::default(), VisibleStepLimit::new(2));
        let state = env.initial_state(&mut rng);
        assert_eq!(env.observe(&state, &mut rng).remaining, 1.0);

        // Step 1
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Continue(_)));
        let state = successor.into_continue().unwrap();
        assert_eq!(env.observe(&state, &mut rng).remaining, 0.5);

        // Step 2
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Interrupt(_)));
        assert_eq!(successor.into_interrupt().unwrap().steps_remaining, 0);
    }
}
