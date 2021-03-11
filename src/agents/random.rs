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

impl<OS: Space, AS: Space> Agent<OS, AS> for RandomAgent<AS> {
    fn act<R: Rng>(
        &mut self,
        observation: &OS::Element,
        _prev_step: Option<Step<OS::Element, AS::Element>>,
        rng: &mut R,
    ) -> AS::Element {
        MarkovAgent::<OS, AS>::act(self, observation, rng)
    }
}

impl<OS: Space, AS: Space> MarkovAgent<OS, AS> for RandomAgent<AS> {
    fn act<R: Rng>(&self, _observation: &OS::Element, rng: &mut R) -> AS::Element {
        self.action_space.sample(rng)
    }

    fn update(&mut self, _step: Step<OS::Element, AS::Element>) {}
}
