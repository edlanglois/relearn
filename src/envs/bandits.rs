use super::{EnvStructure, Environment, StructuredEnvironment};
use crate::spaces::{IndexSpace, Space};
use rand::distributions::{Bernoulli, Distribution};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fmt;

/// A multi-armed bandit with Bernoulli-distribution arm rewards.
pub struct BernoulliBandit {
    probabilities: Vec<f32>,
    rng: StdRng,
}

impl BernoulliBandit {
    pub fn new(probabilities: Vec<f32>, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { probabilities, rng }
    }
}

impl fmt::Display for BernoulliBandit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BernoulliBandit({:?})", self.probabilities)
    }
}

impl Environment for BernoulliBandit {
    type Observation = <IndexSpace as Space>::Element;
    type Action = <IndexSpace as Space>::Element;

    fn step(&mut self, action: &usize) -> (Option<usize>, f32, bool) {
        let prob = self.probabilities[*action];
        let reward = Bernoulli::new(prob as f64).unwrap().sample(&mut self.rng);
        (None, reward as u8 as f32, true)
    }

    fn reset(&mut self) -> usize {
        0
    }
}

impl StructuredEnvironment for BernoulliBandit {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;

    fn structure(&self) -> EnvStructure<IndexSpace, IndexSpace> {
        EnvStructure {
            observation_space: IndexSpace::new(1),
            action_space: IndexSpace::new(self.probabilities.len()),
            reward_range: (0.0, 1.0),
            discount_factor: 1.0,
        }
    }
}
