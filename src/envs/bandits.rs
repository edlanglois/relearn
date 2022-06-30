//! Multi-armed bandit environments
use super::{CloneBuild, EnvDistribution, EnvStructure, Environment, Successor};
use crate::feedback::Reward;
use crate::logging::StatsLogger;
use crate::spaces::{IndexSpace, IntervalSpace, SingletonSpace};
use crate::utils::distributions::{Bernoulli, Bounded, Deterministic, FromMean};
use crate::Prng;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;

/// A multi-armed bandit
///
/// The distribution of each arm has type `D`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bandit<D> {
    distributions: Vec<D>,
}

impl<D: Clone> CloneBuild for Bandit<D> {}

impl<D> Bandit<D> {
    #[must_use]
    pub fn new(distributions: Vec<D>) -> Self {
        Self { distributions }
    }
}

impl<D: Bounded<f64>> EnvStructure for Bandit<D> {
    type ObservationSpace = SingletonSpace;
    type ActionSpace = IndexSpace;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        SingletonSpace::new()
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.distributions.len())
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        let (min, max) = self
            .distributions
            .iter()
            .map(Bounded::bounds)
            .reduce(|(a_min, a_max), (b_min, b_max)| (a_min.min(b_min), a_max.max(b_max)))
            .unwrap_or((0.0, 0.0));
        IntervalSpace::new(Reward(min), Reward(max))
    }

    fn discount_factor(&self) -> f64 {
        1.0
    }
}

impl<D: Distribution<f64> + Bounded<f64>> Environment for Bandit<D> {
    type State = ();
    type Observation = ();
    type Action = usize;
    type Feedback = Reward;

    fn initial_state(&self, _: &mut Prng) -> Self::State {}

    fn observe(&self, _: &Self::State, _: &mut Prng) -> Self::State {}

    fn step(
        &self,
        _state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        _logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        let reward = self.distributions[*action].sample(rng);
        (Successor::Terminate, reward.into())
    }
}

impl<D: FromMean<f64>> Bandit<D> {
    /// Create a new Bandit from a list of means.
    pub fn from_means<I: IntoIterator<Item = T>, T: Borrow<f64>>(
        means: I,
    ) -> Result<Self, D::Error> {
        means
            .into_iter()
            .map(|m| D::from_mean(*m.borrow()))
            .collect::<Result<_, _>>()
            .map(Self::new)
    }
}

/// A multi-armed bandit where each arm samples from a Bernoulli distribution.
pub type BernoulliBandit = Bandit<Bernoulli>;

impl BernoulliBandit {
    /// Create a new `BernoulliBandit` with uniform random means in `[0, 1]`.
    pub fn uniform<R: Rng>(num_arms: usize, rng: &mut R) -> Self {
        let distributions = rng
            .sample_iter(Uniform::new_inclusive(0.0, 1.0))
            .take(num_arms)
            .map(|p| Bernoulli::new(p).unwrap())
            .collect();
        Self { distributions }
    }
}

/// A multi-armed bandit where each arm has a determistic distribution.
pub type DeterministicBandit = Bandit<Deterministic<f64>>;

impl DeterministicBandit {
    /// Create a new `DeterministicBandit` from a list of arm rewards
    pub fn from_values<I: IntoIterator<Item = T>, T: Borrow<f64>>(values: I) -> Self {
        Self::from_means(values).unwrap()
    }
}

/// A distribution over Beroulli bandit environments with uniformly sampled means.
///
/// The mean of each arm is sampled uniformly from `[0, 1]`.
///
/// # Reference
/// This environment distribution is used in the paper
/// "[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning][rl2]" by Duan et al.
///
/// [rl2]: https://arxiv.org/abs/1611.02779
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UniformBernoulliBandits {
    /// Number of bandit arms.
    pub num_arms: usize,
}

impl UniformBernoulliBandits {
    #[must_use]
    pub const fn new(num_arms: usize) -> Self {
        Self { num_arms }
    }
}

impl Default for UniformBernoulliBandits {
    fn default() -> Self {
        Self { num_arms: 2 }
    }
}

impl CloneBuild for UniformBernoulliBandits {}

impl EnvStructure for UniformBernoulliBandits {
    type ObservationSpace = SingletonSpace;
    type ActionSpace = IndexSpace;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        SingletonSpace::new()
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.num_arms)
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        IntervalSpace::new(Reward(0.0), Reward(1.0))
    }

    fn discount_factor(&self) -> f64 {
        1.0
    }
}

impl EnvDistribution for UniformBernoulliBandits {
    type Environment = BernoulliBandit;

    fn sample_environment(&self, rng: &mut Prng) -> Self::Environment {
        BernoulliBandit::uniform(self.num_arms, rng)
    }
}

/// Distribution over deterministic bandits in which one arm has reward 1 and the rest have 0.
///
/// The arm with reward 1 is sampled from a uniform random distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OneHotBandits {
    /// Number of bandit arms.
    pub num_arms: usize,
}

impl OneHotBandits {
    #[must_use]
    pub const fn new(num_arms: usize) -> Self {
        Self { num_arms }
    }
}

impl Default for OneHotBandits {
    fn default() -> Self {
        Self { num_arms: 2 }
    }
}

impl CloneBuild for OneHotBandits {}

impl EnvStructure for OneHotBandits {
    type ObservationSpace = SingletonSpace;
    type ActionSpace = IndexSpace;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        SingletonSpace::new()
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.num_arms)
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        IntervalSpace::new(Reward(0.0), Reward(1.0))
    }

    fn discount_factor(&self) -> f64 {
        1.0
    }
}

impl EnvDistribution for OneHotBandits {
    type Environment = DeterministicBandit;

    fn sample_environment(&self, rng: &mut Prng) -> Self::Environment {
        let mut means = vec![0.0; self.num_arms];
        let index = rng.gen_range(0..self.num_arms);
        means[index] = 1.0;
        DeterministicBandit::from_means(means).unwrap()
    }
}

#[cfg(test)]
mod bernoulli_bandit {
    use super::super::testing;
    use super::*;

    #[test]
    fn run() {
        let env = BernoulliBandit::from_means(vec![0.2, 0.8]).unwrap();
        testing::check_structured_env(&env, 1000, 0);
    }

    #[test]
    fn rewards() {
        let mean = 0.2;
        let num_samples = 10000;
        let env = BernoulliBandit::from_means(vec![mean]).unwrap();
        let mut rng = Prng::seed_from_u64(1);
        let mut reward_1_count = 0;
        for _ in 0..num_samples {
            let (_, feedback) = env.step((), &0, &mut rng, &mut ());
            #[allow(clippy::float_cmp)] // Expecting exact values without error
            if feedback < Reward(0.5) {
                assert_eq!(feedback, Reward(0.0));
            } else {
                assert_eq!(feedback, Reward(1.0));
                reward_1_count += 1
            }
        }
        // Check that the number of 1 rewards is plausible.
        // Approximate the binomial as Gaussian and
        // check that the number of successes is +- 3.5 standard deviations of the mean.
        let bin_mean = f64::from(num_samples) * mean;
        let bin_stddev = (f64::from(num_samples) * mean * (1.0 - mean)).sqrt();
        assert!(
            ((bin_mean - 3.5 * bin_stddev)..=(bin_mean + 3.5 * bin_stddev))
                .contains(&reward_1_count.into())
        );
    }
}

#[cfg(test)]
mod deterministic_bandit {
    use super::super::testing;
    use super::*;

    #[test]
    fn run() {
        let env = DeterministicBandit::from_values(vec![0.2, 0.8]);
        testing::check_structured_env(&env, 1000, 0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Expecting exact values without error
    fn rewards() {
        let mut rng = Prng::seed_from_u64(0);
        let env = DeterministicBandit::from_values(vec![0.2, 0.8]);
        let (_, reward_0) = env.step((), &0, &mut rng, &mut ());
        assert_eq!(reward_0, Reward(0.2));
        let (_, reward_1) = env.step((), &1, &mut rng, &mut ());
        assert_eq!(reward_1, Reward(0.8));
    }
}

#[cfg(test)]
mod uniform_determistic_bandits {
    use super::super::testing;
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_sample() {
        let env_dist = UniformBernoulliBandits::new(3);
        let mut rng = Prng::seed_from_u64(284);
        let env = env_dist.sample_environment(&mut rng);
        testing::check_structured_env(&env, 1000, 286);
    }

    #[test]
    fn subset_env_structure() {
        let env_dist = UniformBernoulliBandits::new(3);
        testing::check_env_distribution_structure(&env_dist, 2);
    }
}

#[cfg(test)]
mod needle_haystack_bandits {
    use super::super::testing;
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_sample() {
        let env_dist = OneHotBandits::new(3);
        let mut rng = Prng::seed_from_u64(284);
        let env = env_dist.sample_environment(&mut rng);
        testing::check_structured_env(&env, 1000, 286);
    }

    #[test]
    fn subset_env_structure() {
        let env_dist = OneHotBandits::new(3);
        testing::check_env_distribution_structure(&env_dist, 2);
    }
}
