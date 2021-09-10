//! Converting a `Pomdp` into an `Environment`
use super::{
    BuildEnv, BuildEnvDist, BuildEnvError, EnvWrapper, Environment, InnerStructureWrapper, Pomdp,
    Wrapped,
};
use crate::envs::EnvStructure;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// A wrapper that adds internal state to an environment.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WithState;

impl<E: Pomdp> EnvWrapper<E> for WithState {
    type Wrapped = PomdpEnv<E>;

    fn wrap<R: Rng + ?Sized>(&self, env: E, rng: &mut R) -> Self::Wrapped {
        PomdpEnv::new(env, rng.gen())
    }
}

impl<E: EnvStructure> InnerStructureWrapper<E> for WithState {}

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

impl<E: Pomdp> IntoStateful for E {
    type Output = PomdpEnv<Self>;

    fn into_stateful(self, seed: u64) -> Self::Output {
        Self::Output::new(self, seed)
    }
}

impl<E: Pomdp, B: BuildEnv<E>> BuildEnv<PomdpEnv<E>> for B {
    fn build_env(&self, seed: u64) -> Result<PomdpEnv<E>, BuildEnvError> {
        // Re-use seed so that BuildEnv<E> and BuildEnv<PomdpEnv<E>>
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
