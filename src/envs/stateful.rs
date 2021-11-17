//! Converting a `Pomdp` into an `Environment`
use super::{Environment, Pomdp};
use crate::envs::EnvStructure;
use crate::logging::Logger;
use rand::{rngs::StdRng, SeedableRng};

/// Wraps a [`Pomdp`] as a [`Environment`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PomdpEnv<E: Pomdp> {
    pub env: E,
    state: Option<E::State>,
    rng: StdRng,
}

impl<E: Pomdp> PomdpEnv<E> {
    pub fn new(env: E, seed: u64) -> Self {
        Self {
            env,
            state: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<E: Pomdp + EnvStructure> EnvStructure for PomdpEnv<E> {
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

impl<E: Pomdp> Environment for PomdpEnv<E> {
    type Observation = E::Observation;
    type Action = E::Action;

    fn step(
        &mut self,
        action: &Self::Action,
        logger: &mut dyn Logger,
    ) -> (Option<Self::Observation>, f64, bool) {
        let state = self
            .state
            .take()
            .expect("Must call reset() before the start of each episode");
        let (next_state, reward, episode_done) =
            self.env.step(state, action, &mut self.rng, logger);
        self.state = next_state;
        let observation = match self.state.as_ref() {
            Some(s) => Some(self.env.observe(s, &mut self.rng)),
            None => None,
        };
        (observation, reward, episode_done)
    }

    fn reset(&mut self) -> Self::Observation {
        let state = self.env.initial_state(&mut self.rng);
        let observation = self.env.observe(&state, &mut self.rng);
        self.state = Some(state);
        observation
    }
}

/// Convert into an [`Environment`].
pub trait IntoEnv {
    type Environment;

    /// Convert into an environment.
    ///
    /// # Args
    /// * `seed` - Seed for pseudo-randomness in the environment state and transition dynamics.
    ///            The environment structure itself should not be random; seed any structural
    ///            randomness from the environment configuration.
    fn into_env(self, seed: u64) -> Self::Environment;
}

impl<E: Pomdp> IntoEnv for E {
    type Environment = PomdpEnv<E>;

    fn into_env(self, seed: u64) -> Self::Environment {
        PomdpEnv::new(self, seed)
    }
}
