//! Environment testing utilities
use super::{
    CloneBuild, DeterministicBandit, EnvDistribution, EnvStructure, Environment,
    StoredEnvStructure, StructuredEnvironment, Successor,
};
use crate::agents::{ActorMode, Agent, RandomAgent};
use crate::simulation::SimSeed;
use crate::spaces::{IndexSpace, SampleSpace, SingletonSpace, Space, SubsetOrd};
use crate::Prng;
use rand::SeedableRng;
use std::cell::Cell;
use std::fmt::Debug;

/// Run a [`StructuredEnvironment`] and check that invariants are satisfied.
///
/// Checks that
/// * Rewards are contained in `env.reward_range()`
/// * `env.discount_factor()` is in the range `[0, 1]`.
/// * Observations are contained in `env.observation_space()`.
/// * Randomly sampled actions from `env.action_space()` are tolerated.
///
pub fn check_structured_env<E>(env: &E, num_steps: usize, seed: u64)
where
    E: StructuredEnvironment,
    E::ObservationSpace: Debug,
    E::Observation: Debug + Clone,
    E::ActionSpace: Debug + SampleSpace + Clone,
    E::Action: Debug,
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

    let agent = RandomAgent::new(action_space);
    env.run(
        Agent::<E::Observation, _>::actor(&agent, ActorMode::Evaluation),
        SimSeed::Root(seed),
        (),
    )
    .take(num_steps)
    .for_each(|step| {
        assert!(step.reward >= min_reward);
        assert!(step.reward <= max_reward);
        assert!(observation_space.contains(&step.observation));
        if let Successor::Interrupt(next_obs) = &step.next {
            assert!(observation_space.contains(next_obs));
        }
    });
}

/// Test that the [`EnvStructure`] of an [`EnvDistribution`] is a superset of its sampled envs.
#[allow(clippy::float_cmp)] // discount factor should be exactly equal
pub fn check_env_distribution_structure<D>(env_dist: &D, num_samples: usize)
where
    D: EnvDistribution + ?Sized,
    D::ObservationSpace: SubsetOrd + Debug,
    D::ActionSpace: SubsetOrd + Debug,
{
    let env_structure = StoredEnvStructure::from(env_dist);
    let (dist_reward_min, dist_reward_max) = env_structure.reward_range;

    let mut rng = Prng::seed_from_u64(75);
    for _ in 0..num_samples {
        let env = env_dist.sample_environment(&mut rng);
        assert!(
            env.observation_space()
                .subset_of(&env_structure.observation_space),
            "{:?} ??? {:?}",
            env.observation_space(),
            env_structure.observation_space,
        );
        assert!(
            env.action_space().subset_of(&env_structure.action_space),
            "{:?} ??? {:?}",
            env.action_space(),
            env_structure.action_space,
        );
        let (env_reward_min, env_reward_max) = env.reward_range();
        assert!(
            dist_reward_min <= env_reward_min,
            "{} </= {}",
            dist_reward_min,
            env_reward_min
        );
        assert!(
            dist_reward_max >= env_reward_max,
            "{} >/= {}",
            dist_reward_max,
            env_reward_max
        );
        assert_eq!(env.discount_factor(), env_structure.discount_factor);
    }
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

impl CloneBuild for RoundRobinDeterministicBandits {}

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

    fn sample_environment(&self, _: &mut Prng) -> Self::Environment {
        let mut values = vec![0.0; self.num_arms];
        let good_arm = self.good_arm.get();
        values[good_arm] = 1.0;
        self.good_arm.set((good_arm + 1) % self.num_arms);
        DeterministicBandit::from_values(&values)
    }
}
