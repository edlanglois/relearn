use super::{Actor, ActorMode, Agent, BatchUpdate, BufferCapacityBound, WriteHistoryBuffer};
use crate::envs::Successor;
use crate::logging::StatsLogger;
use crate::simulation::{PartialStep, Step};
use crate::spaces::FiniteSpace;
use crate::Prng;

/// Wraps an index-space agent as an agent over finite spaces.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FiniteSpaceAgent<T, OS, AS> {
    pub agent: T,
    pub observation_space: OS,
    pub action_space: AS,
}

impl<T, OS, AS> FiniteSpaceAgent<T, OS, AS> {
    pub const fn new(agent: T, observation_space: OS, action_space: AS) -> Self {
        Self {
            agent,
            observation_space,
            action_space,
        }
    }
}

impl<T, OS, AS> Agent<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: Agent<usize, usize>,
    OS: FiniteSpace + Clone + 'static,
    AS: FiniteSpace + Clone + 'static,
    T::HistoryBuffer: 'static,
{
    type Actor = FiniteSpaceActor<T::Actor, OS, AS>;
    fn actor(&self, mode: ActorMode) -> Self::Actor {
        FiniteSpaceActor {
            actor: self.agent.actor(mode),
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
        }
    }
}

/// Wraps an index-space actor as an actor over finite spaces.
pub struct FiniteSpaceActor<T, OS, AS> {
    pub actor: T,
    pub observation_space: OS,
    pub action_space: AS,
}

impl<T, OS, AS> Actor<OS::Element, AS::Element> for FiniteSpaceActor<T, OS, AS>
where
    T: Actor<usize, usize>,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    type EpisodeState = T::EpisodeState;

    fn new_episode_state(&self, rng: &mut Prng) -> Self::EpisodeState {
        self.actor.new_episode_state(rng)
    }

    fn act(
        &self,
        episode_state: &mut Self::EpisodeState,
        observation: &OS::Element,
        rng: &mut Prng,
    ) -> AS::Element {
        let observation_index = self.observation_space.to_index(observation);
        let action_index = self.actor.act(episode_state, &observation_index, rng);
        self.action_space
            .from_index(action_index)
            .expect("invalid action index")
    }
}

impl<T, OS, AS> BatchUpdate<OS::Element, AS::Element> for FiniteSpaceAgent<T, OS, AS>
where
    T: BatchUpdate<usize, usize>,
    OS: FiniteSpace + Clone + 'static,
    AS: FiniteSpace + Clone + 'static,
    T::HistoryBuffer: 'static,
{
    type HistoryBuffer = FiniteSpaceBuffer<T::HistoryBuffer, OS, AS>;

    fn batch_size_hint(&self) -> BufferCapacityBound {
        self.agent.batch_size_hint()
    }

    fn buffer(&self, capacity: BufferCapacityBound) -> Self::HistoryBuffer {
        FiniteSpaceBuffer {
            buffer: self.agent.buffer(capacity),
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
        }
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        self.agent
            .batch_update(buffers.into_iter().map(|b| &mut b.buffer), logger)
    }

    fn batch_update_single(
        &mut self,
        buffer: &mut Self::HistoryBuffer,
        logger: &mut dyn StatsLogger,
    ) {
        self.agent.batch_update_single(&mut buffer.buffer, logger)
    }

    fn batch_update_slice(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn StatsLogger,
    ) {
        self.batch_update(buffers, logger)
    }
}

/// Wraps an index-space buffer to accept finite-space elements.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FiniteSpaceBuffer<B, OS, AS> {
    buffer: B,
    observation_space: OS,
    action_space: AS,
}

impl<B, OS, AS> WriteHistoryBuffer<OS::Element, AS::Element> for FiniteSpaceBuffer<B, OS, AS>
where
    B: WriteHistoryBuffer<usize, usize>,
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn push(&mut self, step: PartialStep<OS::Element, AS::Element>) -> bool {
        self.buffer.push(indexed_partial_step(
            &step,
            &self.observation_space,
            &self.action_space,
        ))
    }

    fn extend_until_ready<I>(&mut self, steps: I) -> bool
    where
        I: IntoIterator<Item = PartialStep<OS::Element, AS::Element>>,
    {
        self.buffer.extend_until_ready(
            steps.into_iter().map(|step| {
                indexed_partial_step(&step, &self.observation_space, &self.action_space)
            }),
        )
    }

    fn clear(&mut self) {
        self.buffer.clear()
    }
}

/// Convert a finite-space `PartialStep` into an index step.
fn indexed_partial_step<OS, AS>(
    step: &PartialStep<OS::Element, AS::Element>,
    observation_space: &OS,
    action_space: &AS,
) -> PartialStep<usize, usize>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    Step {
        observation: observation_space.to_index(&step.observation),
        action: action_space.to_index(&step.action),
        reward: step.reward,
        next: match &step.next {
            Successor::Continue(()) => Successor::Continue(()),
            Successor::Terminate => Successor::Terminate,
            Successor::Interrupt(s) => Successor::Interrupt(observation_space.to_index(s)),
        },
    }
}
