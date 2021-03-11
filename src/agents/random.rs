use super::{Actor, Agent};
use crate::spaces::Space;
use rand::Rng;

/// An agent that always acts randomly
pub struct RandomAgent<AS: Space> {
    action_space: AS,
}

impl<AS: Space> RandomAgent<AS> {
    pub fn new(action_space: AS) -> Self {
        Self { action_space }
    }
}

impl<O, AS: Space, R: Rng> Actor<O, AS::Element, R> for RandomAgent<AS> {
    fn act(&mut self, _observation: &O, _new_episode: bool, rng: &mut R) -> AS::Element {
        self.action_space.sample(rng)
    }
}

impl<O, AS: Space, R: Rng> Agent<O, AS::Element, R> for RandomAgent<AS> {}
