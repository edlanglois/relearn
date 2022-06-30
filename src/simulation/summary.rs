use super::PartialStep;
use crate::feedback::{Feedback, Summary};
use crate::utils::stats::OnlineMeanVariance;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Extend, Sum};
use std::ops::{Add, AddAssign};

/// Summary statistics of simulation steps.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepsSummary<F: Feedback> {
    /// Per-step feedback summary
    pub step_feedback: F::StepSummary,
    /// Per-episode feedback summary
    pub episode_feedback: F::EpisodeSummary,
    /// Episode length statistics
    pub episode_length: OnlineMeanVariance<f64>,
}

impl<F: Feedback> From<OnlineStepsSummary<F>> for StepsSummary<F> {
    #[inline]
    fn from(online_summary: OnlineStepsSummary<F>) -> Self {
        online_summary.completed
    }
}

impl<F: Feedback> Default for StepsSummary<F> {
    fn default() -> Self {
        Self {
            step_feedback: Default::default(),
            episode_feedback: Default::default(),
            episode_length: OnlineMeanVariance::default(),
        }
    }
}

impl<F> fmt::Display for StepsSummary<F>
where
    F: Feedback,
    F::StepSummary: fmt::Display,
    F::EpisodeSummary: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "step_feedback: ")?;
        fmt::Display::fmt(&self.step_feedback, f)?;
        write!(f, "\nepisode_feedback: ")?;
        fmt::Display::fmt(&self.episode_feedback, f)?;
        write!(f, "\nepisode_length: ")?;
        fmt::Display::fmt(&self.episode_length, f)?;
        Ok(())
    }
}

impl<F: Feedback> StepsSummary<F> {
    #[must_use]
    #[inline]
    pub fn num_steps(&self) -> u64 {
        self.step_feedback.size()
    }

    #[must_use]
    #[inline]
    pub fn num_episodes(&self) -> u64 {
        self.episode_feedback.size()
    }
}

impl<'a, O, A, F> FromIterator<&'a PartialStep<O, A, F>> for StepsSummary<F>
where
    O: 'a,
    A: 'a,
    F: Feedback + 'a,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a PartialStep<O, A, F>>,
    {
        OnlineStepsSummary::from_iter(iter).into()
    }
}

impl<O, A, F: Feedback> FromIterator<PartialStep<O, A, F>> for StepsSummary<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = PartialStep<O, A, F>>,
    {
        OnlineStepsSummary::from_iter(iter).into()
    }
}

impl<F: Feedback> Add for StepsSummary<F> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl<F: Feedback> AddAssign for StepsSummary<F> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.step_feedback.merge(other.step_feedback);
        self.episode_feedback.merge(other.episode_feedback);
        self.episode_length += other.episode_length;
    }
}

impl<F: Feedback> Sum for StepsSummary<F> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|a, b| a + b).unwrap_or_default()
    }
}

/// Online calculation of simulation step statistics.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "StepsSummary<F>: Serialize, F::EpisodeFeedback: Serialize",
    deserialize = "StepsSummary<F>: Deserialize<'de>, F::EpisodeFeedback: Deserialize<'de>",
))]
pub struct OnlineStepsSummary<F: Feedback> {
    completed: StepsSummary<F>,
    current_episode_length: u64,
    current_episode_feedback: F::EpisodeFeedback,
}

impl<F> fmt::Debug for OnlineStepsSummary<F>
where
    F: Feedback,
    StepsSummary<F>: fmt::Debug,
    F::EpisodeFeedback: fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("OnlineStepsSummary")
            .field("completed", &self.completed)
            .field("current_episode_length", &self.current_episode_length)
            .field("current_episode_feedback", &self.current_episode_feedback)
            .finish()
    }
}

#[allow(clippy::expl_impl_clone_on_copy)]
impl<F> Clone for OnlineStepsSummary<F>
where
    F: Feedback,
    StepsSummary<F>: Clone,
    F::EpisodeFeedback: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            completed: self.completed.clone(),
            current_episode_length: self.current_episode_length,
            current_episode_feedback: self.current_episode_feedback.clone(),
        }
    }
}

impl<F> Copy for OnlineStepsSummary<F>
where
    F: Feedback,
    StepsSummary<F>: Copy,
    F::EpisodeFeedback: Copy,
{
}

impl<F> PartialEq for OnlineStepsSummary<F>
where
    F: Feedback,
    StepsSummary<F>: PartialEq,
    F::EpisodeFeedback: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.completed == other.completed
            && self.current_episode_length == other.current_episode_length
            && self.current_episode_feedback == other.current_episode_feedback
    }
}

impl<F: Feedback> Default for OnlineStepsSummary<F> {
    fn default() -> Self {
        Self {
            completed: StepsSummary::default(),
            current_episode_length: 0,
            current_episode_feedback: Default::default(),
        }
    }
}

impl<F: Feedback> OnlineStepsSummary<F> {
    pub fn push<O, A>(&mut self, step: &PartialStep<O, A, F>) {
        self.completed.step_feedback.push(&step.feedback);

        self.current_episode_length += 1;
        step.feedback
            .add_to_episode_feedback(&mut self.current_episode_feedback);

        if step.next.episode_done() {
            self.completed
                .episode_feedback
                .push(&self.current_episode_feedback);
            self.current_episode_feedback = Default::default();

            self.completed
                .episode_length
                .push(self.current_episode_length as f64);
            self.current_episode_length = 0;
        }
    }
}

impl<F: Feedback> OnlineStepsSummary<F> {
    #[must_use]
    #[inline]
    pub fn num_steps(&self) -> u64 {
        self.completed.num_steps()
    }

    #[must_use]
    #[inline]
    pub fn num_episodes(&self) -> u64 {
        self.completed.num_episodes()
    }
}

impl<'a, O, A, F> Extend<&'a PartialStep<O, A, F>> for OnlineStepsSummary<F>
where
    O: 'a,
    A: 'a,
    F: Feedback + 'a,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = &'a PartialStep<O, A, F>>,
    {
        for step in iter {
            self.push(step)
        }
    }
}

impl<O, A, F: Feedback> Extend<PartialStep<O, A, F>> for OnlineStepsSummary<F> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = PartialStep<O, A, F>>,
    {
        for step in iter {
            self.push(&step)
        }
    }
}

impl<'a, O, A, F> FromIterator<&'a PartialStep<O, A, F>> for OnlineStepsSummary<F>
where
    O: 'a,
    A: 'a,
    F: Feedback + 'a,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a PartialStep<O, A, F>>,
    {
        let mut s = Self::default();
        s.extend(iter);
        s
    }
}

impl<O, A, F: Feedback> FromIterator<PartialStep<O, A, F>> for OnlineStepsSummary<F> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = PartialStep<O, A, F>>,
    {
        let mut s = Self::default();
        s.extend(iter);
        s
    }
}
