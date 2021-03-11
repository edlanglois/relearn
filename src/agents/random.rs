use super::{Agent, MarkovAgent, Step};
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

impl<O, AS: Space> Agent<O, AS::Element> for RandomAgent<AS> {
    fn act<R: Rng>(
        &mut self,
        observation: &O,
        _prev_step: Option<Step<O, AS::Element>>,
        rng: &mut R,
    ) -> AS::Element {
        MarkovAgent::<O, AS::Element>::act(self, observation, rng)
    }
}

impl<O, AS: Space> MarkovAgent<O, AS::Element> for RandomAgent<AS> {
    fn act<R: Rng>(&self, _observation: &O, rng: &mut R) -> AS::Element {
        self.action_space.sample(rng)
    }

    fn update(&mut self, _step: Step<O, AS::Element>) {}
}
