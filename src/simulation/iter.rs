use super::{PartialStep, TransientStep};
use crate::agents::Actor;
use crate::envs::{Environment, Successor};
use crate::logging::{Event, TimeSeriesLogger};
use std::iter::FusedIterator;
use std::mem;

/// Actor-environment simulation steps.
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

        let mut reset = || {
            self.actor.reset();
            self.environment.reset()
        };

        let (partial_successor, next_observation) = match next {
            Successor::Continue(next_obs) => (Successor::Continue(()), next_obs),
            Successor::Terminate => (Successor::Terminate, reset()),
            Successor::Interrupt(next_obs) => (Successor::Interrupt(next_obs), reset()),
        };
        let observation = mem::replace(&mut self.observation, next_observation);
        let ref_successor = match partial_successor {
            Successor::Continue(()) => Successor::Continue(&self.observation),
            Successor::Terminate => Successor::Terminate,
            Successor::Interrupt(next_obs) => Successor::Interrupt(next_obs),
        };

        TransientStep {
            observation,
            action,
            reward,
            next: ref_successor,
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

    fn next(&mut self) -> Option<Self::Item> {
        let step = self.step();
        Some(PartialStep {
            observation: step.observation,
            action: step.action,
            reward: step.reward,
            next: step.next.into_partial(),
        })
    }

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
