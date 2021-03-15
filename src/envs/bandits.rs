use super::{EnvStructure, Environment, StructuredEnvironment};
use crate::spaces::{IndexSpace, Space};
use rand::distributions::{Bernoulli, Distribution};
use rand::prelude::*;

/// A multi-armed bandit
///
/// The distribution of each arm has type `D`.
#[derive(Debug)]
pub struct Bandit<D: Distribution<f32>> {
    distributions: Vec<D>,
    rng: StdRng,
}

impl<D: Distribution<f32>> Bandit<D> {
    pub fn new(distributions: Vec<D>, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { distributions, rng }
    }
}

impl<D: Distribution<f32>> Environment for Bandit<D> {
    type Observation = <IndexSpace as Space>::Element;
    type Action = <IndexSpace as Space>::Element;

    fn step(&mut self, action: &Self::Action) -> (Option<Self::Observation>, f32, bool) {
        let reward = self.distributions[*action].sample(&mut self.rng);
        (None, reward, true)
    }

    fn reset(&mut self) -> Self::Observation {
        0
    }
}

/// A multi-armed bandit where each arm samples from a Bernoulli distribution.
pub type BernoulliBandit = Bandit<FloatBernoulli>;

impl BernoulliBandit {
    /// Create a new BernoulliBandit from a list of means.
    pub fn from_means(means: Vec<f32>, seed: u64) -> Self {
        Self::new(
            means
                .iter()
                .map(|&p| FloatBernoulli::new(p as f64).unwrap())
                .collect(),
            seed,
        )
    }

    /// Create a new BernoulliBandit with uniform random means.
    pub fn uniform(num_arms: usize, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        let distributions = (0..num_arms)
            .map(|_| FloatBernoulli::new(rand::random()).unwrap())
            .collect();
        Self { distributions, rng }
    }
}

impl StructuredEnvironment for BernoulliBandit {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;

    fn structure(&self) -> EnvStructure<IndexSpace, IndexSpace> {
        EnvStructure {
            observation_space: IndexSpace::new(1),
            action_space: IndexSpace::new(self.distributions.len()),
            reward_range: (0.0, 1.0),
            discount_factor: 1.0,
        }
    }
}

/// Wrapper Bernoulli distribution that can sample floats
pub struct FloatBernoulli(Bernoulli);
impl FloatBernoulli {
    pub fn new(mean: f64) -> Result<FloatBernoulli, rand::distributions::BernoulliError> {
        Ok(Self(Bernoulli::new(mean)?))
    }
}
impl Distribution<f32> for FloatBernoulli {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        self.0.sample(rng) as u8 as f32
    }
}
