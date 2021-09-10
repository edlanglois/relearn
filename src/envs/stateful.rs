//! Converting an `Environment` into a `StatefulEnvironment`
use super::{
    BuildEnvError, BuildEnv, BuildEnvDist, EnvWrapper, Environment, InnerStructureWrapper,
    StatefulEnvironment, Wrapped,
};
use crate::envs::EnvStructure;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A wrapper that adds internal state to an environment.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WithState;

impl<E: Environment> EnvWrapper<E> for WithState {
    type Wrapped = EnvWithState<E>;

    fn wrap<R: Rng + ?Sized>(&self, env: E, rng: &mut R) -> Self::Wrapped {
        EnvWithState::new(env, rng.gen())
    }
}

impl<E: EnvStructure> InnerStructureWrapper<E> for WithState {}

/// Wraps an [`Environment`] as a [`StatefulEnvironment`].
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl<E: Environment + EnvStructure> EnvStructure for EnvWithState<E> {
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
    type Observation = E::Observation;
    type Action = E::Action;

    fn step(&mut self, action: &Self::Action) -> (Option<Self::Observation>, f64, bool) {
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

    fn reset(&mut self) -> Self::Observation {
        let state = self.env.initial_state(&mut self.rng);
        let observation = self.env.observe(&state, &mut self.rng);
        self.state = Some(state);
        observation
    }
}

// TODO: Remove?
/// Supports conversion to a stateful environment
pub trait IntoStateful {
    type Output;

    /// Convert into a stateful environment.
    fn into_stateful(self, seed: u64) -> Self::Output;
}

impl<E: Environment> IntoStateful for E {
    type Output = EnvWithState<Self>;

    fn into_stateful(self, seed: u64) -> Self::Output {
        Self::Output::new(self, seed)
    }
}

impl<E: Environment, B: BuildEnv<E>> BuildEnv<EnvWithState<E>> for B {
    fn build_env(&self, seed: u64) -> Result<EnvWithState<E>, BuildEnvError> {
        // Re-use seed so that BuildEnv<E> and BuildEnv<EnvWithState<E>>
        // have the same environment structure given the same seed.
        let structure_seed = seed;
        // Add an arbitrary offset for the dynamics seed.
        // Want to avoid collissions with other seed derivations.
        let dynamics_seed = seed.wrapping_add(135);
        Ok(self.build_env(structure_seed)?.into_stateful(dynamics_seed))
    }
}

impl<E, B: BuildEnv<E>> BuildEnv<Wrapped<E, WithState>> for B {
    fn build_env(&self, seed: u64) -> Result<Wrapped<E, WithState>, BuildEnvError> {
        Ok(Wrapped {
            inner: self.build_env(seed)?,
            wrapper: WithState,
        })
    }
}

impl<T, B: BuildEnvDist<T>> BuildEnvDist<Wrapped<T, WithState>> for B {
    fn build_env_dist(&self) -> Wrapped<T, WithState> {
        Wrapped {
            inner: self.build_env_dist(),
            wrapper: WithState,
        }
    }
}
