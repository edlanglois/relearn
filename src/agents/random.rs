use super::{Actor, Agent, Step};
use crate::spaces::Space;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fmt;

/// An agent that always acts randomly.
pub struct RandomAgent<AS: Space> {
    action_space: AS,
    rng: StdRng,
}

impl<AS: Space> RandomAgent<AS> {
    pub fn new(action_space: AS, seed: u64) -> Self {
        Self {
            action_space,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<O, AS: Space> Actor<O, AS::Element> for RandomAgent<AS> {
    fn act(&mut self, _observation: &O, _new_episode: bool) -> AS::Element {
        self.action_space.sample(&mut self.rng)
    }
}

impl<O, AS: Space> Agent<O, AS::Element> for RandomAgent<AS> {
    fn update(&mut self, _step: Step<O, AS::Element>) {}
}

impl<AS: Space + fmt::Display> fmt::Display for RandomAgent<AS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RandomAgent({})", self.action_space)
    }
}
