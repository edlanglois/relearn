use super::{EnvSpec, Environment};
use crate::spaces::{IndexSpace, Space};
use rand::distributions::{Bernoulli, Distribution};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A multi-armed bandit with Bernoulli-distribution arm rewards.
pub struct BernoulliBandit {
    probabilities: Vec<f32>,
    rng: StdRng,
}

impl BernoulliBandit {
    pub fn new(probabilities: Vec<f32>, seed: usize) -> Self {
        let rng = StdRng::seed_from_u64(seed as u64);
        Self { probabilities, rng }
    }
}

impl Environment for BernoulliBandit {
    type Observation = <IndexSpace as Space>::Element;
    type Action = <IndexSpace as Space>::Element;

    fn step(&mut self, action: &usize) -> (Option<usize>, f32, bool) {
        let prob = self.probabilities[*action];
        let reward = Bernoulli::new(prob as f64).unwrap().sample(&mut self.rng);
        (None, reward as u8 as f32, false)
    }

    fn reset(&mut self) -> usize {
        0
    }
}

impl EnvSpec for BernoulliBandit {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(0)
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.probabilities.len())
    }

    fn reward_range(&self) -> (f32, f32) {
        (0.0, 1.0)
    }
}
