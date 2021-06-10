use super::{
    BuildEnvError, EnvBuilder, EnvDistBuilder, EnvStructure, EnvWrapper, Environment,
    InnerStructureWrapper, Wrapped,
};
use crate::spaces::Space;
use rand::{rngs::StdRng, Rng};

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

impl<E> EnvWrapper<E> for StepLimit {
    type Wrapped = Wrapped<E, Self>;

    fn wrap<R: Rng + ?Sized>(&self, env: E, _rng: &mut R) -> Self::Wrapped {
        Wrapped::new(env, *self)
    }
}

impl<E: EnvStructure> InnerStructureWrapper<E> for StepLimit {}

impl<E: Environment> Environment for Wrapped<E, StepLimit> {
    /// `(inner_state, current_steps)`
    type State = (E::State, u64);

    fn initial_state(&self, rng: &mut StdRng) -> Self::State {
        (self.inner.initial_state(rng), 0)
    }

    fn observe(
        &self,
        state: &Self::State,
        rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
        self.inner.observe(&state.0, rng)
    }

    fn step(
        &self,
        state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        let (inner_state, mut current_steps) = state;
        let (next_inner_state, reward, mut episode_done) =
            self.inner.step(inner_state, action, rng);
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

impl<B, E> EnvBuilder<Wrapped<E, StepLimit>> for Wrapped<B, StepLimit>
where
    B: EnvBuilder<E>,
{
    fn build_env(&self, seed: u64) -> Result<Wrapped<E, StepLimit>, BuildEnvError> {
        Ok(Wrapped {
            inner: self.inner.build_env(seed)?,
            wrapper: self.wrapper,
        })
    }
}

impl<B, D> EnvDistBuilder<Wrapped<D, StepLimit>> for Wrapped<B, StepLimit>
where
    B: EnvDistBuilder<D>,
{
    fn build_env_dist(&self) -> Wrapped<D, StepLimit> {
        Wrapped {
            inner: self.inner.build_env_dist(),
            wrapper: self.wrapper,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{chain::Move, testing, Chain, EnvBuilder};
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_default() {
        testing::run_stateless(Wrapped::<Chain, StepLimit>::default(), 1000, 104);
    }

    #[test]
    fn build() {
        let config = Wrapped::<Chain, StepLimit>::default();
        let _env: Wrapped<Chain, StepLimit> = config.build_env(0).unwrap();
    }

    #[test]
    fn step_limit() {
        let mut rng = StdRng::seed_from_u64(110);
        let env = Wrapped {
            inner: Chain::default(),
            wrapper: StepLimit::new(2),
        };
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
