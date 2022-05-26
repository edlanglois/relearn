//! Combined actor-agent. Prefer using simulation functions instead.
use super::{Actor, ActorMode, Agent, HistoryDataBound, WriteExperienceIncremental};
use crate::logging::StatsLogger;
use crate::simulation::PartialStep;
use crate::Prng;
use log::info;
use std::fmt;

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
    update_size: HistoryDataBound,
    num_collected_steps: usize,
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
            .field("update_size", &self.update_size)
            .field("num_collected_steps", &self.num_collected_steps)
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
            buffer: agent.buffer(),
            update_size: agent.min_update_size(),
            agent,
            num_collected_steps: 0,
        }
    }

    /// Update with the most recent step result.
    ///
    /// This step must correspond to the most recent call to `Actor::act`.
    pub fn update(&mut self, step: PartialStep<O, A>, logger: &mut dyn StatsLogger) {
        self.num_collected_steps += 1;
        let ready = self
            .update_size
            .is_satisfied(self.num_collected_steps, Some(&step));
        self.buffer
            .write_step(step)
            .unwrap_or_else(|err| info!("error writing step: {}", err));

        if ready {
            self.buffer.end_experience();
            self.actor = None; // Agent cannot be updated while an actor exists.
            self.agent.batch_update_single(&mut self.buffer, logger);
            self.actor = Some(self.agent.actor(ActorMode::Training));
        }
    }
}

impl<T, O, A> Actor<O, A> for SerialActorAgent<T, O, A>
where
    T: Agent<O, A>,
{
    type EpisodeState = <T::Actor as Actor<O, A>>::EpisodeState;

    fn initial_state(&self, rng: &mut Prng) -> Self::EpisodeState {
        self.actor.as_ref().unwrap().initial_state(rng)
    }

    fn act(&self, episode_state: &mut Self::EpisodeState, observation: &O, rng: &mut Prng) -> A {
        self.actor
            .as_ref()
            .unwrap()
            .act(episode_state, observation, rng)
    }
}

// TODO test
