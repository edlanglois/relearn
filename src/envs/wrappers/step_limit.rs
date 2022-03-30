use super::super::{EnvStructure, Environment, Successor};
use super::Wrapped;
use crate::logging::StatsLogger;
use crate::spaces::{IntervalSpace, TupleSpace2};
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

impl<T: EnvStructure> EnvStructure for Wrapped<T, LatentStepLimit> {
    type ObservationSpace = T::ObservationSpace;
    type ActionSpace = T::ActionSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.inner.observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space()
    }

    fn reward_range(&self) -> (f64, f64) {
        self.inner.reward_range()
    }

    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor()
    }
}

impl<E: Environment> Environment for Wrapped<E, LatentStepLimit> {
    /// `(inner_state, step_count)`
    type State = (E::State, u64);
    type Observation = E::Observation;
    type Action = E::Action;

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        (self.inner.initial_state(rng), 0)
    }

    fn observe(&self, state: &Self::State, rng: &mut Prng) -> Self::Observation {
        self.inner.observe(&state.0, rng)
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, f64) {
        let (inner_state, step_count) = state;
        let (inner_successor, reward) = self.inner.step(inner_state, action, rng, logger);

        // Add the step count to the state and interrupt if it is >= max_steps_per_episode
        let successor = match inner_successor.map(|s| (s, step_count + 1)) {
            Successor::Continue((state, steps)) if steps >= self.wrapper.max_steps_per_episode => {
                Successor::Interrupt((state, steps))
            }
            s => s,
        };
        (successor, reward)
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

impl<T: EnvStructure> EnvStructure for Wrapped<T, VisibleStepLimit> {
    type ObservationSpace = TupleSpace2<T::ObservationSpace, IntervalSpace>;
    type ActionSpace = T::ActionSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        TupleSpace2(self.inner.observation_space(), IntervalSpace::new(0.0, 1.0))
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space()
    }

    fn reward_range(&self) -> (f64, f64) {
        self.inner.reward_range()
    }

    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor()
    }
}

impl<E: Environment> Environment for Wrapped<E, VisibleStepLimit> {
    /// `(inner_state, step_count)`
    type State = (E::State, u64);
    type Observation = (E::Observation, f64);
    type Action = E::Action;

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        (self.inner.initial_state(rng), 0)
    }

    fn observe(&self, state: &Self::State, rng: &mut Prng) -> Self::Observation {
        let progress = state.1 as f64 / self.wrapper.max_steps_per_episode as f64;
        (self.inner.observe(&state.0, rng), progress)
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, f64) {
        let (inner_state, step_count) = state;
        let (inner_successor, reward) = self.inner.step(inner_state, action, rng, logger);

        // Add the step count to the state and interrupt if it is >= max_steps_per_episode
        let successor = match inner_successor.map(|s| (s, step_count + 1)) {
            Successor::Continue((state, steps)) if steps >= self.wrapper.max_steps_per_episode => {
                Successor::Interrupt((state, steps))
            }
            s => s,
        };
        (successor, reward)
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
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_default() {
        testing::check_structured_env(&WithVisibleStepLimit::<Chain>::default(), 1000, 119);
    }

    #[test]
    fn build() {
        let config = WithVisibleStepLimit::<Chain>::default();
        let _env = config.build_env(&mut Prng::seed_from_u64(0)).unwrap();
    }

    #[test]
    #[allow(clippy::float_cmp)] // expecting exact values
    fn step_limit() {
        let mut rng = Prng::seed_from_u64(110);
        let env = WithVisibleStepLimit::new(Chain::default(), VisibleStepLimit::new(2));
        let state = env.initial_state(&mut rng);
        assert_eq!(env.observe(&state, &mut rng).1, 0.0);

        // Step 1
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Continue(_)));
        let state = successor.into_continue().unwrap();
        assert_eq!(env.observe(&state, &mut rng).1, 0.5);

        // Step 2
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Interrupt(_)));
    }
}
