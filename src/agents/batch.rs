use super::buffers::{
    BuildHistoryBuffer, HistoryBuffer, SerialBuffer, SerialBufferConfig, WriteHistoryBuffer,
};
use super::{
    Actor, ActorMode, BatchUpdate, BuildAgent, BuildAgentError, BuildBatchAgent, MakeActor,
    PureActor, SetActorMode, SynchronousUpdate,
};
use crate::envs::{EnvStructure, Successor};
use crate::logging::{Event, TimeSeriesLogger};
use crate::simulation::TransientStep;
use crate::spaces::Space;
use std::iter;

/// Configuration for [`SerialBatchAgent`]
pub struct SerialBatchConfig<TC> {
    pub agent_config: TC,
}

impl<TC, OS, AS> BuildAgent<OS, AS> for SerialBatchConfig<TC>
where
    TC: BuildBatchAgent<OS, AS>,
    OS: Space,
    AS: Space,
{
    type Agent = SerialBatchAgent<TC::BatchAgent, OS::Element, AS::Element>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(SerialBatchAgent::new(
            self.agent_config.build_batch_agent(env, seed)?,
        ))
    }
}

/// Wrap a [`BatchUpdate`] as as [`SynchronousUpdate`].
///
/// Caches updates into a history buffer then performs a batch update once full.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SerialBatchAgent<T: BatchUpdate<O, A>, O, A> {
    agent: T,
    history: T::HistoryBuffer,
}

impl<T, O, A> SerialBatchAgent<T, O, A>
where
    T: BatchUpdate<O, A>,
{
    pub fn new(agent: T) -> Self {
        Self {
            history: agent.new_buffer(),
            agent,
        }
    }
}

impl<T, O, A> PureActor<O, A> for SerialBatchAgent<T, O, A>
where
    T: PureActor<O, A> + BatchUpdate<O, A>,
{
    type State = T::State;

    fn initial_state(&self, seed: u64) -> Self::State {
        self.agent.initial_state(seed)
    }
    fn reset_state(&self, state: &mut Self::State) {
        self.agent.reset_state(state)
    }
    fn act(&self, state: &mut Self::State, observation: &O) -> A {
        self.agent.act(state, observation)
    }
}

impl<T, O, A> Actor<O, A> for SerialBatchAgent<T, O, A>
where
    T: Actor<O, A> + BatchUpdate<O, A>,
{
    fn act(&mut self, observation: &O) -> A {
        self.agent.act(observation)
    }
    fn reset(&mut self) {
        self.agent.reset()
    }
}

impl<T, O, A> SynchronousUpdate<O, A> for SerialBatchAgent<T, O, A>
where
    T: BatchUpdate<O, A>,
{
    fn update(&mut self, step: TransientStep<O, A>, logger: &mut dyn TimeSeriesLogger) {
        let full = self.history.push(step.into_partial());
        if full {
            self.agent
                .batch_update(iter::once(&mut self.history), logger);
        }
    }
}

impl<T, O, A> SetActorMode for SerialBatchAgent<T, O, A>
where
    T: BatchUpdate<O, A> + SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.agent.set_actor_mode(mode)
    }
}

/// Marker trait for a [`SynchronousUpdate`] that can accept on-policy updates at any time.
///
/// The updates must still be on-policy and in-order, they just do not have to immediately follow
/// the corresponding call to `SynchronousUpdate::act`.
pub trait AsyncAgent {}

/// Configuration for [`BatchedUpdates`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BatchedUpdatesConfig<TC> {
    pub agent_config: TC,
    pub history_buffer_config: SerialBufferConfig,
}

impl<TC, OS, AS> BuildBatchAgent<OS, AS> for BatchedUpdatesConfig<TC>
where
    TC: BuildAgent<OS, AS>,
    TC::Agent: Actor<OS::Element, AS::Element> + AsyncAgent,
    OS: Space,
    AS: Space,
    // Required because of compiler bug (see impl BatchUpdate for BatchedUpdates)
    OS::Element: 'static,
    AS::Element: 'static,
{
    type BatchAgent = BatchedUpdates<TC::Agent>;

    fn build_batch_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchAgent, BuildAgentError> {
        Ok(BatchedUpdates {
            agent: self.agent_config.build_agent(env, seed)?,
            history_buffer_config: self.history_buffer_config,
        })
    }
}

/// Wrapper that implements [`BatchUpdate`] for a [`SynchronousUpdate`] implementing [`AsyncAgent`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BatchedUpdates<T> {
    agent: T,
    history_buffer_config: SerialBufferConfig,
}

impl<T, O, A> PureActor<O, A> for BatchedUpdates<T>
where
    T: PureActor<O, A>,
{
    type State = T::State;

    fn initial_state(&self, seed: u64) -> Self::State {
        self.agent.initial_state(seed)
    }
    fn reset_state(&self, state: &mut Self::State) {
        self.agent.reset_state(state)
    }
    fn act(&self, state: &mut Self::State, observation: &O) -> A {
        self.agent.act(state, observation)
    }
}

impl<T, O, A> Actor<O, A> for BatchedUpdates<T>
where
    T: Actor<O, A>,
{
    fn act(&mut self, observation: &O) -> A {
        self.agent.act(observation)
    }
    fn reset(&mut self) {
        self.agent.reset()
    }
}

impl<T, O, A> BatchUpdate<O, A> for BatchedUpdates<T>
where
    T: Actor<O, A> + SynchronousUpdate<O, A> + AsyncAgent,
    // Only required because of a compiler bug: https://github.com/rust-lang/rust/issues/85451
    // In `batch_update`, the compiler needs the bound `Self::HistoryBuffer: 'a`
    // for the `I: ...` bound. It wants to infer this bound from `O: 'a` & `A: 'a` and ignores
    // `Self::HistoryBuffer: 'a`.
    O: 'static,
    A: 'static,
{
    type HistoryBuffer = SerialBuffer<O, A>;

    fn new_buffer(&self) -> Self::HistoryBuffer {
        self.history_buffer_config.build_history_buffer()
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn TimeSeriesLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        logger.start_event(Event::AgentOptPeriod).unwrap();
        for buffer in buffers {
            let mut steps = buffer.drain_steps().peekable();
            while let Some(step) = steps.next() {
                let ref_next = match step.next {
                    Successor::Continue(()) => {
                        match steps.peek() {
                            Some(next_step) => Successor::Continue(&next_step.observation),
                            // Next step is missing. Incomplete step so skip this update.
                            // Note: Changing to Terminate would be wrong (return can be non-zero)
                            // and don't have a successor state for Interrupt.
                            None => break,
                        }
                    }
                    Successor::Terminate => Successor::Terminate,
                    Successor::Interrupt(obs) => Successor::Interrupt(obs),
                };
                let transient_step = TransientStep {
                    observation: step.observation,
                    action: step.action,
                    reward: step.reward,
                    next: ref_next,
                };
                self.agent.update(transient_step, logger);
            }
        }

        logger.end_event(Event::AgentOptPeriod).unwrap();
    }
}

impl<'a, T, O, A> MakeActor<'a, O, A> for BatchedUpdates<T>
where
    T: MakeActor<'a, O, A>,
{
    type Actor = T::Actor;
    fn make_actor(&'a self, seed: u64) -> Self::Actor {
        self.agent.make_actor(seed)
    }
}

impl<T> SetActorMode for BatchedUpdates<T>
where
    T: SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.agent.set_actor_mode(mode)
    }
}
