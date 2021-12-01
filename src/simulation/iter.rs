use super::{PartialStep, TransientStep};
use crate::agents::Actor;
use crate::envs::Environment;
use crate::logging::{Event, TimeSeriesLogger};
use std::iter::FusedIterator;

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

    observation: Option<E::Observation>,
}

impl<E, A, L> SimSteps<E, A, L>
where
    E: Environment,
{
    pub fn new(environment: E, actor: A, logger: L) -> Self {
        Self {
            environment,
            actor,
            logger,
            observation: None,
        }
    }
}

impl<E, A, L> SimSteps<E, A, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    L: TimeSeriesLogger,
{
    /// Execute one environment step then evaluate a closure on the resulting state.
    pub fn step_with<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut E, &mut A, TransientStep<E::Observation, E::Action>, &mut L) -> T,
    {
        let observation = match self.observation.take() {
            Some(obs) => obs,
            None => {
                self.actor.reset();
                self.environment.reset()
            }
        };
        let action = self.actor.act(&observation);
        let (next, reward) = self
            .environment
            .step(&action, &mut self.logger.event_logger(Event::EnvStep));

        let (partial_next, next_observation) = next.into_partial_continue();
        self.observation = next_observation;

        let step = TransientStep {
            observation,
            action,
            reward,
            // self.observation is Some(_) in the continue case
            next: partial_next.map_continue(|_| self.observation.as_ref().unwrap()),
        };

        f(
            &mut self.environment,
            &mut self.actor,
            step,
            &mut self.logger,
        )
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
        Some(self.step_with(|_, _, s, _| s.into_partial()))
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
