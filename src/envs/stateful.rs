//! Converting an `Environment` into a `StatefulEnvironment`
use super::{BuildEnvError, EnvBuilder, EnvDistribution, Environment, StatefulEnvironment};
use crate::envs::EnvStructure;
use crate::spaces::Space;
use rand::prelude::*;

/// Creates a [`StatefulEnvironment`] out of an [`Environment`]
#[derive(Debug)]
pub struct EnvWithState<E: Environment> {
    pub env: E,
    state: Option<E::State>,
    rng: StdRng,
}

impl<E: Environment> EnvWithState<E> {
    pub fn new(env: E, seed: u64) -> Self {
        Self {
            env,
            state: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<E: Environment> EnvStructure for EnvWithState<E> {
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

impl<E: Environment> StatefulEnvironment for EnvWithState<E> {
    fn step(
        &mut self,
        action: &<Self::ActionSpace as Space>::Element,
    ) -> (
        Option<<Self::ObservationSpace as Space>::Element>,
        f64,
        bool,
    ) {
        let state = self
            .state
            .take()
            .expect("Must call reset() before the start of each episode");
        let (next_state, reward, episode_done) = self.env.step(state, action, &mut self.rng);
        self.state = next_state;
        let observation = match self.state.as_ref() {
            Some(s) => Some(self.env.observe(s, &mut self.rng)),
            None => None,
        };
        (observation, reward, episode_done)
    }

    fn reset(&mut self) -> <Self::ObservationSpace as Space>::Element {
        let state = self.env.initial_state(&mut self.rng);
        let observation = self.env.observe(&state, &mut self.rng);
        self.state = Some(state);
        observation
    }
}

/// Adds state to the environments of an [`EnvDistribution`].
#[derive(Debug)]
pub struct DistWithState<D>(D);

impl<D> DistWithState<D> {
    pub const fn new(env_dist: D) -> Self {
        Self(env_dist)
    }
}

impl<D: EnvStructure> EnvStructure for DistWithState<D> {
    type ObservationSpace = D::ObservationSpace;
    type ActionSpace = D::ActionSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.0.observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.0.action_space()
    }

    fn reward_range(&self) -> (f64, f64) {
        self.0.reward_range()
    }

    fn discount_factor(&self) -> f64 {
        self.0.discount_factor()
    }
}

impl<D> EnvDistribution for DistWithState<D>
where
    D: EnvDistribution,
    <D as EnvDistribution>::Environment: Environment,
{
    type Environment = EnvWithState<D::Environment>;

    fn sample_environment(&self, rng: &mut StdRng) -> Self::Environment {
        let seed = rng.gen();
        self.0.sample_environment(rng).with_state(seed)
    }
}

/// Supports conversion to a stateful environment
pub trait WithState {
    type Output;

    /// Convert into a stateful environment.
    fn with_state(self, seed: u64) -> Self::Output;
}

impl<E: Environment> WithState for E {
    type Output = EnvWithState<Self>;

    fn with_state(self, seed: u64) -> Self::Output {
        Self::Output::new(self, seed)
    }
}

impl<E: Environment, B: EnvBuilder<E>> EnvBuilder<EnvWithState<E>> for B {
    fn build_env(&self, seed: u64) -> Result<EnvWithState<E>, BuildEnvError> {
        // Re-use seed so that EnvBuilder<E> and EnvBuilder<EnvWithState<E>>
        // have the same environment structure given the same seed.
        let structure_seed = seed;
        // Add an arbitrary offset for the dynamics seed.
        // Want to avoid collissions with other seed derivations.
        let dynamics_seed = seed.wrapping_add(135);
        Ok(self.build_env(structure_seed)?.with_state(dynamics_seed))
    }
}
