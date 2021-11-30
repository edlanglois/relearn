use super::{PartialStep, TransientStep};
use crate::agents::Actor;
use crate::envs::Environment;
use crate::logging::{Event, TimeSeriesLogger};
use std::iter::FusedIterator;
use std::mem;

/// Actor-environment simulation steps.
///
/// Does not perform any updating.
pub struct SimSteps<E, A, L>
where
    E: Environment,
{
    pub environment: E,
    pub actor: A,
    pub logger: L,

    observation: E::Observation,
}

impl<E, A, L> SimSteps<E, A, L>
where
    E: Environment,
{
    pub fn new(mut environment: E, actor: A, logger: L) -> Self {
        let observation = environment.reset();
        Self {
            environment,
            actor,
            logger,
            observation,
        }
    }
}

impl<E, A, L> SimSteps<E, A, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    L: TimeSeriesLogger,
{
    /// Execute one environment step.
    pub fn step(&mut self) -> TransientStep<E::Observation, E::Action> {
        let action = self.actor.act(&self.observation);
        let (next, reward) = self
            .environment
            .step(&action, &mut self.logger.event_logger(Event::EnvStep));

        let (partial_next, next_observation) = next.take_continue_or_else(|| {
            self.actor.reset();
            self.environment.reset()
        });
        let observation = mem::replace(&mut self.observation, next_observation);

        TransientStep {
            observation,
            action,
            reward,
            next: partial_next.map_continue(|_| &self.observation),
        }
    }
}

impl<E, A, L> Iterator for SimSteps<E, A, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    L: TimeSeriesLogger,
{
    /// Cannot return a `TransientStep` without Generic Associated Types
    type Item = PartialStep<E::Observation, E::Action>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step().into_partial())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // infinite
        (usize::MAX, None)
    }
}

impl<E, A, L> FusedIterator for SimSteps<E, A, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    L: TimeSeriesLogger,
{
}
