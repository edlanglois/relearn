use super::{PartialStep, Simulation};
use crate::envs::EnvStructure;
use crate::feedback::Feedback;
use crate::logging::{Loggable, StatsLogger};
use crate::spaces::LogElementSpace;
use serde::{Deserialize, Serialize};
use std::iter::FusedIterator;

/// Simulation steps with logging.
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogSteps<S, OS, AS, EF> {
    steps: S,
    observation_space: OS,
    action_space: AS,
    episode_length: u64,
    episode_feedback: EF,
}

impl<S, OS, AS, EF> LogSteps<S, OS, AS, EF>
where
    S: Simulation,
    S::Environment: EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
    EF: Default,
{
    pub fn new(steps: S) -> Self {
        Self {
            observation_space: steps.env().observation_space(),
            action_space: steps.env().action_space(),
            episode_length: 0,
            episode_feedback: EF::default(),
            steps,
        }
    }
}

impl<S, OS, AS, EF> Simulation for LogSteps<S, OS, AS, EF>
where
    S: Simulation<Observation = OS::Element, Action = AS::Element>,
    S::Feedback: Feedback<EpisodeFeedback = EF>,
    OS: LogElementSpace,
    AS: LogElementSpace,
    EF: Default + Loggable,
{
    type Observation = S::Observation;
    type Action = S::Action;
    type Feedback = S::Feedback;
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

impl<S, OS, AS, EF> Iterator for LogSteps<S, OS, AS, EF>
where
    S: Simulation<Observation = OS::Element, Action = AS::Element>,
    S::Feedback: Feedback<EpisodeFeedback = EF>,
    OS: LogElementSpace,
    AS: LogElementSpace,
    EF: Default + Loggable,
{
    type Item = PartialStep<S::Observation, S::Action, S::Feedback>;

    fn next(&mut self) -> Option<Self::Item> {
        let step = self.steps.next()?;

        let mut group = self.steps.logger_mut().group();
        let mut step_logger = (&mut group).with_scope("step");
        self.observation_space
            .log_element("observation", &step.observation, &mut step_logger)
            .unwrap();
        self.action_space
            .log_element("action", &step.action, &mut step_logger)
            .unwrap();
        step.feedback.log("fbk", &mut step_logger).unwrap();
        step_logger.log_counter_increment("count", 1);

        self.episode_length += 1;
        step.feedback
            .add_to_episode_feedback(&mut self.episode_feedback);

        if step.episode_done() {
            let mut episode_logger = group.with_scope("episode");
            episode_logger.log_scalar("length", self.episode_length as f64);
            self.episode_feedback
                .log("fbk", &mut episode_logger)
                .unwrap();
            episode_logger.log_counter_increment("count", 1);
            self.episode_length = 0;
            self.episode_feedback = Default::default();
        }

        Some(step)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.steps.size_hint()
    }
}

impl<S, OS, AS, EF> ExactSizeIterator for LogSteps<S, OS, AS, EF>
where
    S: ExactSizeIterator + Simulation<Observation = OS::Element, Action = AS::Element>,
    S::Feedback: Feedback<EpisodeFeedback = EF>,
    OS: LogElementSpace,
    AS: LogElementSpace,
    EF: Default + Loggable,
{
    #[inline]
    fn len(&self) -> usize {
        self.steps.len()
    }
}

impl<S, OS, AS, EF> FusedIterator for LogSteps<S, OS, AS, EF>
where
    S: FusedIterator + Simulation<Observation = OS::Element, Action = AS::Element>,
    S::Feedback: Feedback<EpisodeFeedback = EF>,
    OS: LogElementSpace,
    AS: LogElementSpace,
    EF: Default + Loggable,
{
}
