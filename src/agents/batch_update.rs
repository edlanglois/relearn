use super::buffers::{BuildHistoryBuffer, HistoryBuffer};
use super::{Actor, ActorMode, Agent, BuildAgent, BuildAgentError, SetActorMode, Step};
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
    fn batch_update<I: IntoIterator<Item = Step<O, A>>>(
        &mut self,
        steps: I,
        logger: &mut dyn TimeSeriesLogger,
    );
}

/// An agent that accepts updates at any time from any policy.
pub trait OffPolicyAgent {}

impl<T: OffPolicyAgent + Agent<O, A>, O, A> BatchUpdate<O, A> for T {
    fn batch_update<I: IntoIterator<Item = Step<O, A>>>(
        &mut self,
        steps: I,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        for step in steps {
            self.update(step, logger)
        }
        logger.end_event(Event::AgentOptPeriod).unwrap();
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct BatchUpdateAgentConfig<AC, HBC> {
    pub actor_config: AC,
    pub history_buffer_config: HBC,
}

impl<AC, HBC> BatchUpdateAgentConfig<AC, HBC> {
    pub const fn new(actor_config: AC, history_buffer_config: HBC) -> Self {
        Self {
            actor_config,
            history_buffer_config,
        }
    }
}

impl<AC, HBC, OS, AS> BuildAgent<OS, AS> for BatchUpdateAgentConfig<AC, HBC>
where
    AC: BuildBatchUpdateActor<OS, AS>,
    HBC: BuildHistoryBuffer<OS::Element, AS::Element>,
    OS: Space,
    AS: Space,
{
    type Agent = BatchUpdateAgent<AC::BatchUpdateActor, HBC::HistoryBuffer>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        let actor = self.actor_config.build_batch_update_actor(env, seed)?;
        let history = self.history_buffer_config.build_history_buffer();
        Ok(BatchUpdateAgent { actor, history })
    }
}

/// Wrapper that implements an agent for `T: [Actor] + [BatchUpdate]`.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct BatchUpdateAgent<T, HB> {
    actor: T,
    history: HB,
}

impl<O, A, T, HB> Actor<O, A> for BatchUpdateAgent<T, HB>
where
    T: Actor<O, A>,
{
    fn act(&mut self, observation: &O, new_episode: bool) -> A {
        self.actor.act(observation, new_episode)
    }
}

impl<O, A, T, HB> Agent<O, A> for BatchUpdateAgent<T, HB>
where
    T: Actor<O, A> + BatchUpdate<O, A>,
    HB: HistoryBuffer<O, A>,
{
    fn update(&mut self, step: Step<O, A>, logger: &mut dyn TimeSeriesLogger) {
        let full = self.history.push(step);
        if full {
            self.actor.batch_update(self.history.drain_steps(), logger);
        }
    }
}

impl<T, HB> SetActorMode for BatchUpdateAgent<T, HB>
where
    T: SetActorMode,
{
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.actor.set_actor_mode(mode)
    }
}

#[cfg(test)]
mod batch_tabular_q_learning {
    use super::super::{
        history::EpisodeBufferConfig, testing, BuildAgent, TabularQLearningAgentConfig,
    };
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        let config = BatchUpdateAgentConfig::new(
            TabularQLearningAgentConfig::default(),
            EpisodeBufferConfig {
                ep_done_step_threshold: 20,
                step_threshold: 25,
            },
        );
        testing::train_deterministic_bandit(|env| config.build_agent(env, 0).unwrap(), 1000, 0.9);
    }
}
