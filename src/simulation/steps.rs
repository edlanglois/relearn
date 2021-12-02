use super::{PartialStep, TransientStep};
use crate::agents::Actor;
use crate::envs::Environment;
use crate::logging::{Event, TimeSeriesLogger};
use std::fmt;
use std::iter::FusedIterator;

/// Iterator of actor-environment simulation steps.
pub struct ActorSteps<E, A, L>
where
    E: Environment,
{
    pub environment: E,
    pub actor: A,
    pub logger: L,

    observation: Option<E::Observation>,
}

impl<E, A, L> ActorSteps<E, A, L>
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

impl<E, A, L> ActorSteps<E, A, L>
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

impl<E, A, L> Iterator for ActorSteps<E, A, L>
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

impl<E, A, L> FusedIterator for ActorSteps<E, A, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    L: TimeSeriesLogger,
{
}

/// Basic summary statistics of a simulation
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct SimulationSummary {
    pub num_steps: u64,
    pub num_episodes: u64,
    pub total_reward: f64,
}
impl fmt::Display for SimulationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "num_steps: {}", self.num_steps)?;
        writeln!(f, "num_episodes: {}", self.num_episodes)?;
        writeln!(
            f,
            "mean_step_reward: {}",
            self.total_reward / self.num_steps as f64
        )?;
        writeln!(
            f,
            "mean_ep_reward:   {}",
            self.total_reward / self.num_episodes as f64
        )?;
        writeln!(
            f,
            "mean_ep_length:   {}",
            self.num_steps as f64 / self.num_episodes as f64
        )?;
        Ok(())
    }
}

// TODO: Make this a trait
impl SimulationSummary {
    pub fn update<O, A>(&mut self, step: &PartialStep<O, A>) {
        self.num_steps += 1;
        self.total_reward += step.reward;
        if step.next.episode_done() {
            self.num_episodes += 1;
        }
    }

    pub fn from_steps<I: Iterator<Item = PartialStep<O, A>>, O, A>(steps: I) -> Self {
        steps.fold(Self::default(), |mut s, step| {
            s.update(&step);
            s
        })
    }
}
