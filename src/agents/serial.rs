//! Combined actor-agent. Prefer using simulation functions instead.
use super::{Actor, ActorMode, Agent, WriteHistoryBuffer};
use crate::logging::StatsLogger;
use crate::simulation::PartialStep;
use crate::Prng;
use std::fmt;
use std::iter;

/// A serial combined actor-agent.
///
/// Consists of an agent, an actor, and a history buffer.
/// Steps are accumulated into the buffer and used to update the agent when full.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SerialActorAgent<T, O, A>
where
    T: Agent<O, A>,
{
    agent: T,
    actor: Option<T::Actor>,
    buffer: T::HistoryBuffer,
}

// Avoid depending on O: Debug & A: Debug
impl<T, O, A> fmt::Debug for SerialActorAgent<T, O, A>
where
    T: Agent<O, A> + fmt::Debug,
    T::Actor: fmt::Debug,
    T::HistoryBuffer: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SerialActorAgent")
            .field("agent", &self.agent)
            .field("actor", &self.actor)
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl<T, O, A> SerialActorAgent<T, O, A>
where
    T: Agent<O, A>,
{
    pub fn new(agent: T) -> Self {
        Self {
            actor: Some(agent.actor(ActorMode::Training)),
            buffer: agent.buffer(agent.batch_size_hint()),
            agent,
        }
    }

    /// Update with the most recent step result.
    ///
    /// This step must correspond to the most recent call to `Actor::act`.
    pub fn update(&mut self, step: PartialStep<O, A>, logger: &mut dyn StatsLogger) {
        let full = self.buffer.push(step);
        if full {
            self.actor = None; // Agent cannot be updated while an actor exists.
            self.agent
                .batch_update(iter::once(&mut self.buffer), logger);
            self.actor = Some(self.agent.actor(ActorMode::Training));
        }
    }
}

impl<T, O, A> Actor<O, A> for SerialActorAgent<T, O, A>
where
    T: Agent<O, A>,
{
    type EpisodeState = <T::Actor as Actor<O, A>>::EpisodeState;

    fn new_episode_state(&self, rng: &mut Prng) -> Self::EpisodeState {
        self.actor.as_ref().unwrap().new_episode_state(rng)
    }

    fn act(&self, episode_state: &mut Self::EpisodeState, observation: &O, rng: &mut Prng) -> A {
        self.actor
            .as_ref()
            .unwrap()
            .act(episode_state, observation, rng)
    }
}

// TODO test
