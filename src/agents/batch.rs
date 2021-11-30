use super::buffers::{BuildHistoryBuffer, HistoryBuffer, SerialBuffer, SerialBufferConfig};
use super::{
    Actor, ActorMode, Agent, BuildAgent, BuildAgentError, FullStep, SetActorMode, SyncParams,
    SyncParamsError,
};
use crate::envs::EnvStructure;
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::Space;

/// Build an actor supporting batch updates ([`BatchUpdate`]).
pub trait BuildBatchUpdateActor<OS: Space, AS: Space> {
    type BatchUpdateActor: Actor<OS::Element, AS::Element>
        + BatchUpdate<OS::Element, AS::Element>
        + SetActorMode;

    /// Build an actor for the given environment structure ([`EnvStructure`]).
    ///
    /// The agent is built in [`ActorMode::Training`].
    ///
    /// # Args
    /// * `env`  - The structure of the environment in which the agent is to operate.
    /// * `seed` - A number used to seed the agent's random state,
    ///            for those agents that use deterministic pseudo-random number generation.
    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError>;
}

/// An agent that can update from a batch of on-policy history steps.
pub trait BatchUpdate<O, A> {
    fn batch_update(
        &mut self,
        history: &mut dyn HistoryBuffer<O, A>,
        logger: &mut dyn TimeSeriesLogger,
    );
}

/// Marker trait for agents that can accept updates at any time from any policy.
pub trait OffPolicyAgent<O, A>: Agent<O, A> {}

/// Helper implementaiton of [`BatchUpdate::batch_update`] for [`OffPolicyAgent`].
pub fn off_policy_batch_update<T, O, A>(
    agent: &mut T,
    history: &mut dyn HistoryBuffer<O, A>,
    logger: &mut dyn TimeSeriesLogger,
) where
    T: OffPolicyAgent<O, A>,
{
    logger.start_event(Event::AgentOptPeriod).unwrap();
    for step in history.drain_steps() {
        agent.update(step, logger)
    }
    logger.end_event(Event::AgentOptPeriod).unwrap();
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct BatchUpdateAgentConfig<AC> {
    pub actor_config: AC,
    pub history_buffer_config: SerialBufferConfig,
}

impl<AC> BatchUpdateAgentConfig<AC> {
    pub const fn new(actor_config: AC, history_buffer_config: SerialBufferConfig) -> Self {
        Self {
            actor_config,
            history_buffer_config,
        }
    }
}

impl<AC, OS, AS> BuildAgent<OS, AS> for BatchUpdateAgentConfig<AC>
where
    AC: BuildBatchUpdateActor<OS, AS>,
    OS: Space,
    AS: Space,
{
    type Agent = BatchUpdateAgent<AC::BatchUpdateActor, OS::Element, AS::Element>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        BatchUpdateAgent::new(&self.actor_config, &self.history_buffer_config, env, seed)
    }
}

/// Wrapper that implements an agent for `T: [Actor] + [BatchUpdate]`.
#[derive(Debug, Clone)]
pub struct BatchUpdateAgent<T, O, A> {
    actor: T,
    history: SerialBuffer<O, A>,
}

impl<T, O, A> BatchUpdateAgent<T, O, A> {
    pub fn new<TC, OS, AS>(
        actor_config: &TC,
        history_buffer_config: &SerialBufferConfig,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self, BuildAgentError>
    where
        TC: BuildBatchUpdateActor<OS, AS, BatchUpdateActor = T> + ?Sized,
        OS: Space<Element = O>,
        AS: Space<Element = A>,
    {
        let actor = actor_config.build_batch_update_actor(env, seed)?;
        let history = history_buffer_config.build_history_buffer();
        Ok(Self { actor, history })
    }
}

impl<T, O, A> Actor<O, A> for BatchUpdateAgent<T, O, A>
where
    T: Actor<O, A>,
{
    fn act(&mut self, observation: &O, new_episode: bool) -> A {
        self.actor.act(observation, new_episode)
    }
}

impl<T, O, A> Agent<O, A> for BatchUpdateAgent<T, O, A>
where
    T: Actor<O, A> + BatchUpdate<O, A>,
{
    fn update(&mut self, step: FullStep<O, A>, logger: &mut dyn TimeSeriesLogger) {
        let full = self.history.push(step);
        if full {
            self.actor.batch_update(&mut self.history, logger);
            self.history.clear();
        }
    }
}

impl<T, O, A> SetActorMode for BatchUpdateAgent<T, O, A>
where
    T: SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.actor.set_actor_mode(mode)
    }
}

impl<T, O, A> SyncParams for BatchUpdateAgent<T, O, A>
where
    T: SyncParams,
{
    fn sync_params(&mut self, target: &Self) -> Result<(), SyncParamsError> {
        self.actor.sync_params(&target.actor)
    }
}

#[cfg(test)]
mod batch_tabular_q_learning {
    use super::super::{
        buffers::SerialBufferConfig, testing, BuildAgent, TabularQLearningAgentConfig,
    };
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        let config = BatchUpdateAgentConfig::new(
            TabularQLearningAgentConfig::default(),
            SerialBufferConfig {
                soft_threshold: 20,
                hard_threshold: 25,
            },
        );
        testing::train_deterministic_bandit(|env| config.build_agent(env, 0).unwrap(), 1000, 0.9);
    }
}
