use super::{
    buffers::NullBuffer, Actor, ActorMode, Agent, BatchUpdate, BufferCapacityBound, BuildAgent,
    BuildAgentError,
};
use crate::envs::EnvStructure;
use crate::logging::StatsLogger;
use crate::spaces::{SampleSpace, Space};
use crate::Prng;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Configuration for a random agent.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RandomAgentConfig;

impl RandomAgentConfig {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl<OS, AS> BuildAgent<OS, AS> for RandomAgentConfig
where
    OS: Space,
    AS: SampleSpace + Clone,
{
    type Agent = RandomAgent<AS>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(RandomAgent::new(env.action_space()))
    }
}

/// An agent that samples actions at random.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RandomAgent<AS> {
    action_space: AS,
}

impl<AS> RandomAgent<AS> {
    pub const fn new(action_space: AS) -> Self {
        Self { action_space }
    }
}

impl<AS: fmt::Display> fmt::Display for RandomAgent<AS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RandomAgent({})", self.action_space)
    }
}

impl<O, AS: SampleSpace + Clone> Agent<O, AS::Element> for RandomAgent<AS> {
    type Actor = Self;
    fn actor(&self, _mode: ActorMode) -> Self::Actor {
        self.clone()
    }
}

impl<O, AS: SampleSpace + Clone> Actor<O, AS::Element> for RandomAgent<AS> {
    type EpisodeState = ();

    fn new_episode_state(&self, _: &mut Prng) -> Self::EpisodeState {}

    fn act(&self, _: &mut Self::EpisodeState, _: &O, rng: &mut Prng) -> AS::Element {
        self.action_space.sample(rng)
    }
}

impl<O, AS: Space> BatchUpdate<O, AS::Element> for RandomAgent<AS> {
    type HistoryBuffer = NullBuffer;

    fn batch_size_hint(&self) -> BufferCapacityBound {
        BufferCapacityBound::empty()
    }

    fn buffer(&self, _: BufferCapacityBound) -> Self::HistoryBuffer {
        NullBuffer
    }

    fn batch_update<'a, I>(&mut self, _buffers: I, _logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
    }

    fn batch_update_single(
        &mut self,
        _buffer: &mut Self::HistoryBuffer,
        _logger: &mut dyn StatsLogger,
    ) {
    }

    fn batch_update_slice(
        &mut self,
        _buffers: &mut [Self::HistoryBuffer],
        _logger: &mut dyn StatsLogger,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::{DeterministicBandit, EnvStructure, Environment};
    use crate::simulation::SimSeed;
    use rand::SeedableRng;

    #[test]
    /// Check that actions are contained in env action space.
    fn actions_are_legal() {
        let env = DeterministicBandit::from_values([0.0, 1.0, 0.5]);
        let agent = RandomAgentConfig
            .build_agent(&env, &mut Prng::seed_from_u64(116))
            .unwrap();
        let actor = Agent::<<DeterministicBandit as EnvStructure>::ObservationSpace, _>::actor(
            &agent,
            ActorMode::Training,
        );
        let action_space = env.action_space();

        let steps = env.run(&actor, SimSeed::Root(117), ());
        for step in steps.take(1000) {
            assert!(action_space.contains(&step.action));
        }
    }
}
