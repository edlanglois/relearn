use super::{BuildEnvError, EnvBuilder, EnvStructure, Environment};
use crate::spaces::Space;
use rand::rngs::StdRng;

/// Environment wrapper that cuts off episodes after a set number of steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepLimit<E> {
    /// Inner environment
    pub env: E,
    /// Maximum number of steps per episode
    pub max_steps_per_episode: u64,
}

impl<E> StepLimit<E> {
    pub const fn new(env: E, max_steps_per_episode: u64) -> Self {
        Self {
            env,
            max_steps_per_episode,
        }
    }
}

impl<E: Default> Default for StepLimit<E> {
    fn default() -> Self {
        Self {
            env: E::default(),
            max_steps_per_episode: 100,
        }
    }
}

/// `StepLimit` can also be used as a configurion
impl<E: EnvBuilder<T>, T> EnvBuilder<StepLimit<T>> for StepLimit<E> {
    fn build_env(&self, seed: u64) -> Result<StepLimit<T>, BuildEnvError> {
        Ok(StepLimit {
            env: self.env.build_env(seed)?,
            max_steps_per_episode: self.max_steps_per_episode,
        })
    }
}

impl<E: EnvStructure> EnvStructure for StepLimit<E> {
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;
    fn observation_space(&self) -> Self::ObservationSpace {
        self.env.observation_space()
    }
    fn action_space(&self) -> Self::ActionSpace {
        self.env.action_space()
    }
    fn reward_range(&self) -> (f64, f64) {
        self.env.reward_range()
    }
    fn discount_factor(&self) -> f64 {
        self.env.discount_factor()
    }
}

impl<E: Environment> Environment for StepLimit<E> {
    /// `(inner_state, current_steps)`
    type State = (E::State, u64);

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        (self.env.initial_state(rng), 0)
    }

    fn observe(
        &self,
        state: &Self::State,
        rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
        self.env.observe(&state.0, rng)
    }

    fn step(
        &self,
        state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        let (inner_state, mut current_steps) = state;
        let (next_inner_state, reward, mut episode_done) = self.env.step(inner_state, action, rng);
        current_steps += 1;

        // Attach the new current step count to the state
        let next_state = next_inner_state.map(|s| (s, current_steps));

        // Check if the step limit has been reached.
        // If so, cut off the episode (but don't mark next_state as terminal)
        if current_steps >= self.max_steps_per_episode {
            episode_done = true;
        }
        (next_state, reward, episode_done)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{chain::Move, testing, Chain};
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_default() {
        testing::run_stateless(StepLimit::<Chain>::default(), 1000, 104);
    }

    #[test]
    fn step_limit() {
        let mut rng = StdRng::seed_from_u64(110);
        let env = StepLimit::new(Chain::default(), 2);
        let state = env.initial_state(&mut rng);

        // Step 1
        let (opt_state, _, episode_done) = env.step(state, &Move::Left, &mut rng);
        assert!(!episode_done);
        let state = opt_state.unwrap();

        // Step 2
        let (state, _, episode_done) = env.step(state, &Move::Left, &mut rng);
        assert!(episode_done);
        assert!(state.is_some());
    }
}
