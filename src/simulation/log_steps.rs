use super::{PartialStep, Simulation};
use crate::envs::EnvStructure;
use crate::logging::{LogValue, StatsLogger};
use crate::spaces::ElementRefInto;
use serde::{Deserialize, Serialize};
use std::iter::FusedIterator;

/// Simulation steps with logging.
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogSteps<S, OS, AS> {
    steps: S,
    episode_reward: f64,
    episode_length: u64,
    observation_space: OS,
    action_space: AS,
}

impl<S, OS, AS> LogSteps<S, OS, AS>
where
    S: Simulation,
    S::Environment: EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
{
    pub fn new(steps: S) -> Self {
        Self {
            episode_reward: 0.0,
            episode_length: 0,
            observation_space: steps.env().observation_space(),
            action_space: steps.env().action_space(),
            steps,
        }
    }
}

impl<S, OS, AS> Simulation for LogSteps<S, OS, AS>
where
    S: Simulation<Observation = OS::Element, Action = AS::Element>,
    OS: ElementRefInto<LogValue>,
    AS: ElementRefInto<LogValue>,
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

impl<S, OS, AS> Iterator for LogSteps<S, OS, AS>
where
    S: Simulation<Observation = OS::Element, Action = AS::Element>,
    OS: ElementRefInto<LogValue>,
    AS: ElementRefInto<LogValue>,
{
    type Item = PartialStep<S::Observation, S::Action>;

    fn next(&mut self) -> Option<Self::Item> {
        let step = self.steps.next()?;

        let mut group = self.steps.logger_mut().group();
        let mut step_logger = (&mut group).with_scope("step");
        step_logger.log_scalar("reward", step.reward);
        step_logger
            .log(
                "observation".into(),
                self.observation_space.elem_ref_into(&step.observation),
            )
            .unwrap();
        step_logger
            .log(
                "action".into(),
                self.action_space.elem_ref_into(&step.action),
            )
            .unwrap();

        step_logger.log_counter_increment("count", 1);
        self.episode_reward += step.reward;
        self.episode_length += 1;
        if step.episode_done() {
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

impl<S, OS, AS> ExactSizeIterator for LogSteps<S, OS, AS>
where
    S: ExactSizeIterator + Simulation<Observation = OS::Element, Action = AS::Element>,
    OS: ElementRefInto<LogValue>,
    AS: ElementRefInto<LogValue>,
{
    #[inline]
    fn len(&self) -> usize {
        self.steps.len()
    }
}

impl<S, OS, AS> FusedIterator for LogSteps<S, OS, AS>
where
    S: FusedIterator + Simulation<Observation = OS::Element, Action = AS::Element>,
    OS: ElementRefInto<LogValue>,
    AS: ElementRefInto<LogValue>,
{
}
