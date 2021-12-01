use super::{Actor, BuildAgent, BuildAgentError, SetActorMode, SynchronousAgent};
use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::simulation::TransientStep;
use crate::spaces::{SampleSpace, Space};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fmt;

/// Configuration setttings for a random agent.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RandomAgentConfig;

impl RandomAgentConfig {
    pub const fn new() -> Self {
        Self
    }
}

impl<OS, AS> BuildAgent<OS, AS> for RandomAgentConfig
where
    OS: Space,
    AS: SampleSpace,
{
    type Agent = RandomAgent<AS>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(RandomAgent::new(env.action_space(), seed))
    }
}

/// An agent that always acts randomly.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    fn act(&mut self, _observation: &O) -> AS::Element {
        self.action_space.sample(&mut self.rng)
    }

    fn reset(&mut self) {
        // No episode state
    }
}

impl<O, AS: SampleSpace> SynchronousAgent<O, AS::Element> for RandomAgent<AS> {
    fn update(&mut self, _step: TransientStep<O, AS::Element>, _logger: &mut dyn TimeSeriesLogger) {
    }
}

impl<AS: fmt::Display> fmt::Display for RandomAgent<AS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RandomAgent({})", self.action_space)
    }
}

impl<AS> SetActorMode for RandomAgent<AS> {}
