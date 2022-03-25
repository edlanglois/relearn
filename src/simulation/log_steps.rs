use super::{PartialStep, Simulation};
use crate::logging::StatsLogger;
use serde::{Deserialize, Serialize};
use std::iter::FusedIterator;

/// Simulation steps with logging.
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogSteps<S> {
    steps: S,
    episode_reward: f64,
    episode_length: u64,
}

impl<S> LogSteps<S> {
    pub const fn new(steps: S) -> Self {
        Self {
            steps,
            episode_reward: 0.0,
            episode_length: 0,
        }
    }
}

impl<S> Simulation for LogSteps<S>
where
    S: Simulation,
{
    type Observation = S::Observation;
    type Action = S::Action;
    type Environment = S::Environment;
    type Actor = S::Actor;
    type Logger = S::Logger;

    #[inline]
    fn env(&self) -> &Self::Environment {
        self.steps.env()
    }
    #[inline]
    fn env_mut(&mut self) -> &mut Self::Environment {
        self.steps.env_mut()
    }
    #[inline]
    fn actor(&self) -> &Self::Actor {
        self.steps.actor()
    }
    #[inline]
    fn actor_mut(&mut self) -> &mut Self::Actor {
        self.steps.actor_mut()
    }
    #[inline]
    fn logger(&self) -> &Self::Logger {
        self.steps.logger()
    }
    #[inline]
    fn logger_mut(&mut self) -> &mut Self::Logger {
        self.steps.logger_mut()
    }
}

impl<S> Iterator for LogSteps<S>
where
    S: Simulation,
{
    type Item = PartialStep<S::Observation, S::Action>;

    fn next(&mut self) -> Option<Self::Item> {
        let step = self.steps.next()?;

        let mut group = self.steps.logger_mut().group();
        let mut step_logger = (&mut group).with_scope("step");
        step_logger.log_scalar("reward", step.reward);
        // TODO: Log action and observation
        step_logger.log_counter_increment("count", 1);
        self.episode_reward += step.reward;
        self.episode_length += 1;
        if step.next.episode_done() {
            let mut episode_logger = group.with_scope("episode");
            episode_logger.log_scalar("reward", self.episode_reward);
            episode_logger.log_scalar("length", self.episode_length as f64);
            episode_logger.log_counter_increment("count", 1);
            self.episode_reward = 0.0;
            self.episode_length = 0;
        }

        Some(step)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.steps.size_hint()
    }
}

impl<S> ExactSizeIterator for LogSteps<S>
where
    S: Simulation + ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.steps.len()
    }
}

impl<S> FusedIterator for LogSteps<S> where S: Simulation + FusedIterator {}
