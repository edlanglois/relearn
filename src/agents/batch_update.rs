use super::history::HistoryBuffer;
use super::{Actor, Agent, Step};
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
        logger.end_event(Event::AgentOptPeriod);
    }
}

/// Wrapper that implements an agent for `T: [Actor] + [BatchUpdate]`.
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
