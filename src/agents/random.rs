use super::{AsyncUpdate, BuildAgent, BuildAgentError, PureActor, SetActorMode, SynchronousUpdate};
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
        _seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(RandomAgent::new(env.action_space()))
    }
}

/// An agent that always acts randomly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomAgent<AS> {
    action_space: AS,
}

impl<AS> RandomAgent<AS> {
    pub const fn new(action_space: AS) -> Self {
        Self { action_space }
    }
}

impl<O, AS: SampleSpace> PureActor<O, AS::Element> for RandomAgent<AS> {
    type State = StdRng;

    fn initial_state(&self, seed: u64) -> Self::State {
        StdRng::seed_from_u64(seed)
    }

    fn reset_state(&self, _state: &mut Self::State) {}

    fn act(&self, rng: &mut Self::State, _observation: &O) -> AS::Element {
        self.action_space.sample(rng)
    }
}

impl<O, AS: Space> SynchronousUpdate<O, AS::Element> for RandomAgent<AS> {
    fn update(&mut self, _step: TransientStep<O, AS::Element>, _logger: &mut dyn TimeSeriesLogger) {
    }
}

impl<AS> AsyncUpdate for RandomAgent<AS> {}

impl<AS: fmt::Display> fmt::Display for RandomAgent<AS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RandomAgent({})", self.action_space)
    }
}

impl<AS> SetActorMode for RandomAgent<AS> {}
