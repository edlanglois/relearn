//! Multi-armed bandit environments
use super::{BuildEnvError, EnvBuilder, EnvStructure, Environment};
use crate::spaces::{IndexSpace, SingletonSpace, Space};
use crate::utils::distributions::{Bernoulli, Bounded, Deterministic, FromMean};
use rand::distributions::{Distribution, Standard};
use rand::prelude::*;
use std::borrow::Borrow;

/// Configuration for a Bandit environment with fixed means
#[derive(Debug)]
pub struct FixedMeansBanditConfig {
    /// The arm means
    pub means: Vec<f64>,
}

impl Default for FixedMeansBanditConfig {
    fn default() -> Self {
        FixedMeansBanditConfig {
            means: vec![0.2, 0.8],
        }
    }
}

impl<D> EnvBuilder<Bandit<D>> for FixedMeansBanditConfig
where
    D: Distribution<f64> + FromMean<f64>,
    <D as FromMean<f64>>::Error: Into<BuildEnvError>,
{
    fn build_env(&self, _seed: u64) -> Result<Bandit<D>, BuildEnvError> {
        Bandit::from_means(&self.means).map_err(|e: <D as FromMean<f64>>::Error| e.into())
    }
}
/// Configuration for a Bandit environment with arm means drawn IID from some distribution.
#[derive(Debug)]
pub struct PriorMeansBanditConfig<D> {
    /// The number of arms
    pub num_arms: usize,
    /// The arm mean prior distribution
    pub mean_prior: D,
}

impl Default for PriorMeansBanditConfig<Standard> {
    fn default() -> Self {
        Self {
            num_arms: 10,
            mean_prior: Standard,
        }
    }
}

impl<DR, DM> EnvBuilder<Bandit<DR>> for PriorMeansBanditConfig<DM>
where
    DR: Distribution<f64> + FromMean<f64>,
    <DR as FromMean<f64>>::Error: Into<BuildEnvError>,
    DM: Distribution<f64>,
{
    fn build_env(&self, seed: u64) -> Result<Bandit<DR>, BuildEnvError> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mean_prior = &self.mean_prior;
        Bandit::from_means(
            (0..self.num_arms)
                .into_iter()
                .map(|_| mean_prior.sample(&mut rng)),
        )
        .map_err(|e: <DR as FromMean<f64>>::Error| e.into())
    }
}

/// A multi-armed bandit
///
/// The distribution of each arm has type `D`.
#[derive(Debug)]
pub struct Bandit<D> {
    distributions: Vec<D>,
}

impl<D> Bandit<D> {
    pub fn new(distributions: Vec<D>) -> Self {
        Self { distributions }
    }
}

impl<D: Distribution<f64> + Bounded<f64>> Environment for Bandit<D> {
    type State = ();
    type ObservationSpace = SingletonSpace;
    type ActionSpace = IndexSpace;

    fn initial_state(&self, _rng: &mut StdRng) -> Self::State {}

    fn observe(
        &self,
        _state: &Self::State,
        _rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
    }

    fn step(
        &self,
        _state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        let reward = self.distributions[*action].sample(rng);
        (None, reward, true)
    }

    fn structure(&self) -> EnvStructure<Self::ObservationSpace, Self::ActionSpace> {
        let reward_range = self.distributions.iter().map(|d| d.bounds()).fold(
            (0.0, -1.0),
            |(a_min, a_max), (b_min, b_max)| {
                if a_max < a_min {
                    (b_min, b_max)
                } else if b_max < b_min {
                    (a_min, a_max)
                } else {
                    (a_min.min(b_min), a_max.max(b_max))
                }
            },
        );
        EnvStructure {
            observation_space: SingletonSpace::new(),
            action_space: IndexSpace::new(self.distributions.len()),
            reward_range,
            discount_factor: 1.0,
        }
    }
}

impl<D: FromMean<f64>> Bandit<D> {
    /// Create a new Bandit from a list of means.
    pub fn from_means<I: IntoIterator<Item = T>, T: Borrow<f64>>(
        means: I,
    ) -> Result<Self, <D as FromMean<f64>>::Error> {
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
    /// Create a new BernoulliBandit with uniform random means.
    pub fn uniform<R: Rng>(num_arms: u32, rng: &mut R) -> Self {
        let distributions = (0..num_arms)
            .map(|_| Bernoulli::new(rng.gen()).unwrap())
            .collect();
        Self { distributions }
    }
}

/// A multi-armed bandit where each arm has a determistic distribution.
pub type DeterministicBandit = Bandit<Deterministic<f64>>;

impl DeterministicBandit {
    /// Create a new DeterministicBandit from a list of arm rewards
    pub fn from_values<I: IntoIterator<Item = T>, T: Borrow<f64>>(values: I) -> Self {
        Self::from_means(values).unwrap()
    }
}

#[cfg(test)]
mod bernoulli_bandit {
    use super::super::testing;
    use super::*;

    #[test]
    fn run() {
        let env = BernoulliBandit::from_means(vec![0.2, 0.8]).unwrap();
        testing::run_stateless(env, 1000, 0);
    }

    #[test]
    fn rewards() {
        let mean = 0.2;
        let num_samples = 10000;
        let env = BernoulliBandit::from_means(vec![mean]).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let mut reward_1_count = 0;
        for _ in 0..num_samples {
            let (_, reward, _) = env.step((), &0, &mut rng);
            #[allow(clippy::float_cmp)] // Expecting exact values without error
            if reward < 0.5 {
                assert_eq!(reward, 0.0);
            } else {
                assert_eq!(reward, 1.0);
                reward_1_count += 1
            }
        }
        // Check that the number of 1 rewards is plausible.
        // Approximate the binomial as Gaussian and
        // check that the number of successes is +- 3 standard deviations of the mean.
        let bin_mean = (num_samples as f64) * mean;
        let bin_stddev = ((num_samples as f64) * mean * (1.0 - mean)).sqrt();
        assert!((reward_1_count as f64) > bin_mean - 3.0 * bin_stddev);
        assert!((reward_1_count as f64) < bin_mean + 3.0 * bin_stddev);
    }
}

#[cfg(test)]
mod deterministic_bandit {
    use super::super::testing;
    use super::*;

    #[test]
    fn run() {
        let env = DeterministicBandit::from_values(vec![0.2, 0.8]);
        testing::run_stateless(env, 1000, 0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Expecting exact values without error
    fn rewards() {
        let mut rng = StdRng::seed_from_u64(0);
        let env = DeterministicBandit::from_values(vec![0.2, 0.8]);
        let (_, reward, _) = env.step((), &0, &mut rng);
        assert_eq!(reward, 0.2);
        let (_, reward, _) = env.step((), &1, &mut rng);
        assert_eq!(reward, 0.8);
    }
}
