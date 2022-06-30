use super::{Feedback, Summary};
use crate::logging::{LogError, Loggable, StatsLogger};

impl<A: Loggable, B: Loggable> Loggable for (A, B) {
    fn log<L: StatsLogger + ?Sized>(
        &self,
        name: &'static str,
        logger: &mut L,
    ) -> Result<(), LogError> {
        let mut logger = logger.group().with_scope(name);
        let r0 = self.0.log("0", &mut logger);
        let r1 = self.1.log("1", &mut logger);
        r0.and(r1)
    }
}

impl<A: Summary, B: Summary> Summary for (A, B) {
    type Item = (A::Item, B::Item);

    fn push(&mut self, item: &Self::Item) {
        self.0.push(&item.0);
        self.1.push(&item.1);
    }
    fn size(&self) -> u64 {
        self.0.size()
    }
    fn merge(&mut self, other: Self) {
        self.0.merge(other.0);
        self.1.merge(other.1);
    }
}

impl<A: Feedback, B: Feedback> Feedback for (A, B) {
    type EpisodeFeedback = (A::EpisodeFeedback, B::EpisodeFeedback);
    type StepSummary = (A::StepSummary, B::StepSummary);
    type EpisodeSummary = (A::EpisodeSummary, B::EpisodeSummary);

    fn add_to_episode_feedback(&self, episode_feedback: &mut Self::EpisodeFeedback) {
        self.0.add_to_episode_feedback(&mut episode_feedback.0);
        self.1.add_to_episode_feedback(&mut episode_feedback.1);
    }
}
