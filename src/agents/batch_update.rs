use super::history::HistoryBuffer;
use super::{Actor, ActorMode, Agent, AgentBuilder, BuildAgentError, SetActorMode, Step};
use crate::logging::{Event, TimeSeriesLogger};

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
pub struct BatchUpdateAgentConfig<B, HBB> {
    pub actor_config: B,
    pub history_buffer_config: HBB,
}

impl<T, B, HB, HBB, E> AgentBuilder<BatchUpdateAgent<T, HB>, E> for BatchUpdateAgentConfig<B, HBB>
where
    B: AgentBuilder<T, E>,
    HB: for<'a> From<&'a HBB>,
    E: ?Sized,
{
    fn build_agent(&self, env: &E, seed: u64) -> Result<BatchUpdateAgent<T, HB>, BuildAgentError> {
        let actor = self.actor_config.build_agent(env, seed)?;
        let history = (&self.history_buffer_config).into();
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
        history::{EpisodeBuffer, EpisodeBufferConfig},
        testing, TabularQLearningAgent, TabularQLearningAgentConfig,
    };
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        let config = BatchUpdateAgentConfig {
            actor_config: TabularQLearningAgentConfig::default(),
            history_buffer_config: EpisodeBufferConfig {
                ep_done_step_threshold: 20,
                step_threshold: 25,
            },
        };
        testing::train_deterministic_bandit(
            |env_structure| -> BatchUpdateAgent<TabularQLearningAgent<_, _>, EpisodeBuffer<_, _>> {
                config.build_agent(env_structure, 0).unwrap()
            },
            1000,
            0.9,
        );
    }
}
