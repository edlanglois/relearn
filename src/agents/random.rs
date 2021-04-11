use super::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use crate::envs::EnvStructure;
use crate::logging::Logger;
use crate::spaces::SampleSpace;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fmt;

/// Configuration setttings for a random agent.
#[derive(Debug)]
pub struct RandomAgentConfig;

impl RandomAgentConfig {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RandomAgentConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<OS, AS> AgentBuilder<RandomAgent<AS>, OS, AS> for RandomAgentConfig {
    fn build(
        &self,
        es: EnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<RandomAgent<AS>, BuildAgentError> {
        Ok(RandomAgent::new(es.action_space, seed))
    }
}

/// An agent that always acts randomly.
pub struct RandomAgent<AS> {
    action_space: AS,
    rng: StdRng,
}

impl<AS> RandomAgent<AS> {
    pub fn new(action_space: AS, seed: u64) -> Self {
        Self {
            action_space,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<O, AS: SampleSpace> Actor<O, AS::Element> for RandomAgent<AS> {
    fn act(&mut self, _observation: &O, _new_episode: bool) -> AS::Element {
        self.action_space.sample(&mut self.rng)
    }
}

impl<O, AS: SampleSpace> Agent<O, AS::Element> for RandomAgent<AS> {
    fn act(&mut self, observation: &O, new_episode: bool) -> AS::Element {
        Actor::act(self, observation, new_episode)
    }

    fn update(&mut self, _step: Step<O, AS::Element>, _logger: &mut dyn Logger) {}
}

impl<AS: fmt::Display> fmt::Display for RandomAgent<AS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RandomAgent({})", self.action_space)
    }
}
