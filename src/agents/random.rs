use super::{Actor, Agent};
use crate::spaces::Space;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// An agent that always acts randomly
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

impl<O, AS: Space> Agent<O, AS::Element> for RandomAgent<AS> {}
