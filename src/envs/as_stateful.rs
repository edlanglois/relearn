//! Converting an Environment into a StatefulEnvironment
use super::{Environment, StatefulEnvironment};
use crate::envs::EnvStructure;
use crate::spaces::Space;
use rand::prelude::*;

/// Creates a StatefulEnvironment out of an Environment
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

impl<E: Environment> StatefulEnvironment for EnvWithState<E> {
    type ObservationSpace = E::ObservationSpace;
    type ActionSpace = E::ActionSpace;

    fn step(
        &mut self,
        action: &<Self::ActionSpace as Space>::Element,
    ) -> (
        Option<<Self::ObservationSpace as Space>::Element>,
        f32,
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

    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace> {
        self.env.structure()
    }
}

/// Supports conversion to a stateful environment
pub trait AsStateful {
    type Output: StatefulEnvironment;

    /// Convert into a stateful environment.
    fn as_stateful(self, seed: u64) -> Self::Output;
}

impl<E: Environment> AsStateful for E {
    type Output = EnvWithState<E>;

    fn as_stateful(self, seed: u64) -> Self::Output {
        Self::Output::new(self, seed)
    }
}
