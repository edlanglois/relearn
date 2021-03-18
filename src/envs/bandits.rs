use super::{EnvStructure, Environment};
use crate::spaces::{IndexSpace, Space};
use rand::distributions::{Bernoulli, Distribution};
use rand::prelude::*;

/// A multi-armed bandit
///
/// The distribution of each arm has type `D`.
#[derive(Debug)]
pub struct Bandit<D: Distribution<f32>> {
    distributions: Vec<D>,
}

impl<D: Distribution<f32>> Bandit<D> {
    pub fn new(distributions: Vec<D>) -> Self {
        Self { distributions }
    }
}

pub struct Empty {}

impl<D: Distribution<f32>> Environment for Bandit<D> {
    type State = Empty;
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;

    fn initial_state(&self, _rng: &mut StdRng) -> Self::State {
        Empty {}
    }

    fn observe(
        &self,
        _state: &Self::State,
        _rng: &mut StdRng,
    ) -> <Self::ObservationSpace as Space>::Element {
        0
    }

    fn step(
        &self,
        _state: Self::State,
        action: &<Self::ActionSpace as Space>::Element,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f32, bool) {
        let reward = self.distributions[*action].sample(rng);
        (None, reward, true)
    }

    fn structure(&self) -> EnvStructure<IndexSpace, IndexSpace> {
        EnvStructure {
            observation_space: IndexSpace::new(1),
            action_space: IndexSpace::new(self.distributions.len()),
            reward_range: (0.0, 1.0),
            discount_factor: 1.0,
        }
    }
}

/// A multi-armed bandit where each arm samples from a Bernoulli distribution.
pub type BernoulliBandit = Bandit<FloatBernoulli>;

impl BernoulliBandit {
    /// Create a new BernoulliBandit from a list of means.
    pub fn from_means(means: Vec<f32>) -> Self {
        Self::new(
            means
                .iter()
                .map(|&p| FloatBernoulli::new(p as f64).unwrap())
                .collect(),
        )
    }

    /// Create a new BernoulliBandit with uniform random means.
    pub fn uniform<R: Rng>(num_arms: usize, rng: &mut R) -> Self {
        let distributions = (0..num_arms)
            .map(|_| FloatBernoulli::new(rng.gen()).unwrap())
            .collect();
        Self { distributions }
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
