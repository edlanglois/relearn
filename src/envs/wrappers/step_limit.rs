use super::super::{Environment, Successor};
use super::Wrapped;
use crate::logging::StatsLogger;
use crate::Prng;

/// Environment wrapper that cuts off episodes after a set number of steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepLimit {
    /// Maximum number of steps per episode
    pub max_steps_per_episode: u64,
}

impl StepLimit {
    pub const fn new(max_steps_per_episode: u64) -> Self {
        Self {
            max_steps_per_episode,
        }
    }
}

impl Default for StepLimit {
    fn default() -> Self {
        Self {
            max_steps_per_episode: 100,
        }
    }
}

/// Wrap an environment with a per-episode step limit.
pub type WithStepLimit<E> = Wrapped<E, StepLimit>;

impl<E: Environment> Environment for Wrapped<E, StepLimit> {
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

#[cfg(test)]
mod tests {
    use super::super::super::{chain::Move, testing, BuildEnv, Chain};
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_default() {
        testing::check_structured_env(&WithStepLimit::<Chain>::default(), 1000, 119);
    }

    #[test]
    fn build() {
        let config = WithStepLimit::<Chain>::default();
        let _env = config.build_env(&mut Prng::seed_from_u64(0)).unwrap();
    }

    #[test]
    fn step_limit() {
        let mut rng = Prng::seed_from_u64(110);
        let env = WithStepLimit::new(Chain::default(), StepLimit::new(2));
        let state = env.initial_state(&mut rng);

        // Step 1
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Continue(_)));
        let state = successor.continue_().unwrap();

        // Step 2
        let (successor, _) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(matches!(successor, Successor::Interrupt(_)));
    }
}
