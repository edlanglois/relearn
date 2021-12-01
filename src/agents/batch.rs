use super::{
    Actor, ActorMode, BatchAgent, BuildAgent, BuildAgentError, SetActorMode, SynchronousAgent,
    WriteHistoryBuffer,
};
use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::simulation::TransientStep;
use crate::spaces::Space;
use std::slice;

/// Configuration for [`SerialBatchAgent`]
pub struct SerialBatchConfig<TC> {
    pub agent_config: TC,
}

impl<TC, OS, AS> BuildAgent<OS, AS> for SerialBatchConfig<TC>
where
    TC: BuildAgent<OS, AS>,
    TC::Agent: BatchAgent<OS::Element, AS::Element>,
    OS: Space,
    AS: Space,
{
    type Agent = SerialBatchAgent<TC::Agent, OS::Element, AS::Element>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(self.agent_config.build_agent(env, seed)?.into())
    }
}

/// Wrap a [`BatchAgent`] as as [`SynchronousAgent`].
///
/// Caches updates into a history buffer then performs a batch update once full.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SerialBatchAgent<T: BatchAgent<O, A>, O, A> {
    agent: T,
    history: T::HistoryBuffer,
}

impl<T, O, A> SerialBatchAgent<T, O, A>
where
    T: BatchAgent<O, A>,
{
    pub fn new(agent: T) -> Self {
        let history = agent.new_buffer();
        Self { agent, history }
    }
}

impl<T, O, A> From<T> for SerialBatchAgent<T, O, A>
where
    T: BatchAgent<O, A>,
{
    fn from(agent: T) -> Self {
        Self::new(agent)
    }
}

impl<T, O, A> Actor<O, A> for SerialBatchAgent<T, O, A>
where
    T: BatchAgent<O, A>,
{
    fn act(&mut self, observation: &O) -> A {
        self.agent.act(observation)
    }
    fn reset(&mut self) {
        self.agent.reset()
    }
}

impl<T, O, A> SynchronousAgent<O, A> for SerialBatchAgent<T, O, A>
where
    T: BatchAgent<O, A>,
{
    fn update(&mut self, step: TransientStep<O, A>, logger: &mut dyn TimeSeriesLogger) {
        let full = self.history.push(step.into_partial());
        if full {
            self.agent
                .batch_update(slice::from_ref(&self.history), logger);
        }
        self.history.clear();
    }
}

impl<T, O, A> SetActorMode for SerialBatchAgent<T, O, A>
where
    T: BatchAgent<O, A> + SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.agent.set_actor_mode(mode)
    }
}
