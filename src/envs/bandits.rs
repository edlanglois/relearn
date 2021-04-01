//! Multi-armed bandit environments
use super::{EnvStructure, Environment};
use crate::spaces::{IndexSpace, SingletonSpace, Space};
use rand::distributions::{Bernoulli, Distribution};
use rand::prelude::*;

/// A multi-armed bandit
///
/// The distribution of each arm has type `D`.
#[derive(Debug)]
pub struct Bandit<D: Distribution<f64>> {
    distributions: Vec<D>,
}

impl<D: Distribution<f64>> Bandit<D> {
    pub fn new(distributions: Vec<D>) -> Self {
        Self { distributions }
    }
}

impl<D: Distribution<f64> + Bounded> Environment for Bandit<D> {
    type State = ();
    type ObservationSpace = SingletonSpace;
    type ActionSpace = IndexSpace;

    fn initial_state(&self, _rng: &mut StdRng) -> Self::State {
        ()
    }

    fn observe(
        &self,
        _state: &Self::State,
        _rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
        ()
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
            reward_range: reward_range,
            discount_factor: 1.0,
        }
    }
}

/// A multi-armed bandit where each arm samples from a Bernoulli distribution.
pub type BernoulliBandit = Bandit<FloatBernoulli>;

impl BernoulliBandit {
    /// Create a new BernoulliBandit from a list of means.
    pub fn from_means<I: IntoIterator<Item = f64>>(
        means: I,
    ) -> Result<Self, rand::distributions::BernoulliError> {
        let distributions = means
            .into_iter()
            .map(|p| FloatBernoulli::new(p as f64))
            .collect::<Result<_, _>>()?;
        Ok(Self::new(distributions))
    }

    /// Create a new BernoulliBandit with uniform random means.
    pub fn uniform<R: Rng>(num_arms: u32, rng: &mut R) -> Self {
        let distributions = (0..num_arms)
            .map(|_| FloatBernoulli::new(rng.gen()).unwrap())
            .collect();
        Self { distributions }
    }
}

/// Wrapper Bernoulli distribution that can sample floats
#[derive(Debug)]
pub struct FloatBernoulli(Bernoulli);
impl FloatBernoulli {
    pub fn new(mean: f64) -> Result<FloatBernoulli, rand::distributions::BernoulliError> {
        Ok(Self(Bernoulli::new(mean)?))
    }
}
impl Distribution<f64> for FloatBernoulli {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.0.sample(rng) as u8 as f64
    }
}
impl Bounded for FloatBernoulli {
    fn bounds(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
}

/// A multi-armed bandit where each arm has a determistic distribution.
pub type DeterministicBandit = Bandit<Deterministic>;

impl DeterministicBandit {
    /// Create a new DeterministicBandit from a list of values.
    pub fn from_values<I: IntoIterator<Item = f64>>(means: I) -> Self {
        let distributions = means.into_iter().map(Deterministic::new).collect();
        Self::new(distributions)
    }
}

/// A determistic distribution.
///
/// Always produces the same value when sampled.
#[derive(Debug)]
pub struct Deterministic(f64);

impl Deterministic {
    pub fn new(value: f64) -> Self {
        Self(value)
    }
}

impl Distribution<f64> for Deterministic {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        self.0
    }
}
impl Bounded for Deterministic {
    fn bounds(&self) -> (f64, f64) {
        (self.0, self.0)
    }
}

/// Bounds on a scalar value
pub trait Bounded {
    /// Minimum and maximum values (inclusive). Infinities are allowed
    ///
    /// If max < min then the interval is empty.
    fn bounds(&self) -> (f64, f64);
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
    fn rewards() {
        let mut rng = StdRng::seed_from_u64(0);
        let env = DeterministicBandit::from_values(vec![0.2, 0.8]);
        let (_, reward, _) = env.step((), &0, &mut rng);
        assert_eq!(reward, 0.2);
        let (_, reward, _) = env.step((), &1, &mut rng);
        assert_eq!(reward, 0.8);
    }
}
