//! Environment testing utilities
use super::{
    DeterministicBandit, EnvDistribution, EnvStructure, Environment, StatefulEnvironment, WithState,
};
use crate::agents::{RandomAgent, Step};
use crate::simulation;
use crate::simulation::hooks::{ClosureHook, StepLimit};
use crate::spaces::{IndexSpace, SampleSpace, SingletonSpace, Space};
use rand::rngs::StdRng;
use std::cell::Cell;
use std::fmt::Debug;

/// Run a stateless environment and check that invariants are satisfied.
pub fn run_stateless<E>(env: E, num_steps: u64, seed: u64)
where
    E: Environment,
    <E as EnvStructure>::ObservationSpace: Debug,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Debug + Clone,
    <E as EnvStructure>::ActionSpace: Debug + SampleSpace,
    <<E as EnvStructure>::ActionSpace as Space>::Element: Debug,
{
    run_stateful(&mut env.with_state(seed), num_steps, seed + 1)
}

/// Run a stateful environment and check that invariants are satisfied.
pub fn run_stateful<E>(env: &mut E, num_steps: u64, seed: u64)
where
    E: StatefulEnvironment,
    <E as EnvStructure>::ObservationSpace: Debug,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Debug + Clone,
    <E as EnvStructure>::ActionSpace: Debug + SampleSpace,
    <<E as EnvStructure>::ActionSpace as Space>::Element: Debug,
{
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let (min_reward, max_reward) = env.reward_range();
    assert!(min_reward <= max_reward);

    let discount_factor = env.discount_factor();
    assert!(discount_factor >= 0.0);
    assert!(discount_factor <= 1.0);

    if num_steps == 0 {
        return;
    }

    let mut agent = RandomAgent::new(action_space, seed);
    simulation::run_agent(
        env,
        &mut agent,
        &mut (),
        &mut (
            ClosureHook::from(|step: &Step<_, _>| -> bool {
                assert!(step.reward >= min_reward);
                assert!(step.reward <= max_reward);
                if let Some(obs) = &step.next_observation {
                    assert!(observation_space.contains(obs))
                } else {
                    assert!(step.episode_done);
                }
                true
            }),
            StepLimit::new(num_steps),
        ),
    );
}

/// Deterministic multi-armed bandit "distribution" with a different goal arm on each sample.
///
/// The first environment sampled has reward 1 only on the first arm and 0s on the rest,
/// the second environment sampled has reward 1 only on the second arm, etc.
/// Wraps around to the first arm upon reaching the end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundRobinDeterministicBandits {
    pub num_arms: usize,
    good_arm: Cell<usize>,
}

impl RoundRobinDeterministicBandits {
    pub const fn new(num_arms: usize) -> Self {
        Self {
            num_arms,
            good_arm: Cell::new(0),
        }
    }
}

impl EnvStructure for RoundRobinDeterministicBandits {
    type ObservationSpace = SingletonSpace;
    type ActionSpace = IndexSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        SingletonSpace::new()
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.num_arms)
    }

    fn reward_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    fn discount_factor(&self) -> f64 {
        1.0
    }
}

impl EnvDistribution for RoundRobinDeterministicBandits {
    type Environment = DeterministicBandit;

    fn sample_environment(&self, _rng: &mut StdRng) -> Self::Environment {
        let mut values = vec![0.0; self.num_arms];
        let good_arm = self.good_arm.get();
        values[good_arm] = 1.0;
        self.good_arm.set((good_arm + 1) % self.num_arms);
        DeterministicBandit::from_values(&values)
    }
}
