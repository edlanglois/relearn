//! Agent-environment feedback
mod reward;
mod tuple;

pub use reward::Reward;

use crate::logging::Loggable;

pub trait Feedback: Loggable + Clone {
    // TODO: Instead of `Default`, use a constructor.
    // The construct should maybe be a method of a `FeedbackSpace` trait?
    // TODO: Support discount factor.
    // Discount factor could be part of a `RewardSpace` but not necessarily other feedback spaces?
    /// Overall feedback for an episode.
    type EpisodeFeedback: Default + Loggable;
    /// Per-step feedback summary.
    type StepSummary: Summary<Item = Self> + Loggable;
    /// Per-episode feedback summary.
    type EpisodeSummary: Summary<Item = Self::EpisodeFeedback> + Loggable;

    fn add_to_episode_feedback(&self, episode_feedback: &mut Self::EpisodeFeedback);
}

/// Summarizes a collection of items.
pub trait Summary: Default {
    type Item;
    /// Add an item to the summary.
    fn push(&mut self, item: &Self::Item);
    /// Number of items added to the summary
    fn size(&self) -> u64;
    /// Merge with another summary
    fn merge(&mut self, other: Self);
}
