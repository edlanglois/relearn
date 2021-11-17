use super::super::Pomdp;
use super::Wrapped;
use crate::logging::Logger;
use rand::rngs::StdRng;

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

impl<E: Pomdp> Pomdp for Wrapped<E, StepLimit> {
    /// `(inner_state, current_steps)`
    type State = (E::State, u64);
    type Observation = E::Observation;
    type Action = E::Action;

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        (self.inner.initial_state(rng), 0)
    }

    fn observe(&self, state: &Self::State, rng: &mut StdRng) -> Self::Observation {
        self.inner.observe(&state.0, rng)
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
        logger: &mut dyn Logger,
    ) -> (Option<Self::State>, f64, bool) {
        let (inner_state, mut current_steps) = state;
        let (next_inner_state, reward, mut episode_done) =
            self.inner.step(inner_state, action, rng, logger);
        current_steps += 1;

        // Attach the new current step count to the state
        let next_state = next_inner_state.map(|s| (s, current_steps));

        // Check if the step limit has been reached.
        // If so, cut off the episode (but don't mark next_state as terminal)
        if current_steps >= self.wrapper.max_steps_per_episode {
            episode_done = true;
        }
        (next_state, reward, episode_done)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{chain::Move, testing, BuildEnv, Chain, PomdpEnv};
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_default() {
        testing::run_pomdp(WithStepLimit::<Chain>::default(), 1000, 119);
    }

    #[test]
    fn build() {
        let config = WithStepLimit::<Chain>::default();
        let _env: PomdpEnv<WithStepLimit<Chain>> = config.build_env(0).unwrap();
    }

    #[test]
    fn step_limit() {
        let mut rng = StdRng::seed_from_u64(110);
        let env = WithStepLimit::new(Chain::default(), StepLimit::new(2));
        let state = env.initial_state(&mut rng);

        // Step 1
        let (opt_state, _, episode_done) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(!episode_done);
        let state = opt_state.unwrap();

        // Step 2
        let (state, _, episode_done) = env.step(state, &Move::Left, &mut rng, &mut ());
        assert!(episode_done);
        assert!(state.is_some());
    }
}
