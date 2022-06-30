use super::{Feedback, Summary};
use crate::logging::{LogError, LogValue, Loggable, StatsLogger};
use crate::utils::stats::OnlineMeanVariance;
use derive_more::{Add, AddAssign, Sub, SubAssign};
use num_traits::{Bounded, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A reward-signal environment feedback
#[derive(
    Debug,
    Default,
    Copy,
    Clone,
    PartialEq,
    PartialOrd,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Serialize,
    Deserialize,
)]
pub struct Reward(pub f64);

impl fmt::Display for Reward {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl From<f64> for Reward {
    #[inline]
    fn from(r: f64) -> Self {
        Self(r)
    }
}

impl From<Reward> for f64 {
    #[inline]
    fn from(Reward(r): Reward) -> Self {
        r
    }
}

impl Reward {
    /// Get the reward value as a float
    #[inline]
    #[must_use]
    pub const fn unwrap(self) -> f64 {
        self.0
    }
}

impl Feedback for Reward {
    type EpisodeFeedback = Self;
    type StepSummary = RewardSummary;
    type EpisodeSummary = RewardSummary;
    #[inline]
    fn add_to_episode_feedback(&self, episode_feedback: &mut Self::EpisodeFeedback) {
        episode_feedback.0 += self.0
    }
}

impl Bounded for Reward {
    #[inline]
    fn min_value() -> Self {
        Self(Bounded::min_value())
    }
    #[inline]
    fn max_value() -> Self {
        Self(Bounded::max_value())
    }
}

impl ToPrimitive for Reward {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

impl Loggable for Reward {
    fn log<L: StatsLogger + ?Sized>(
        &self,
        name: &'static str,
        logger: &mut L,
    ) -> Result<(), LogError> {
        logger
            .with_scope(name)
            .log("reward".into(), LogValue::Scalar(self.0))
    }
}

/// A summmary of reward values.
#[derive(Debug, Default, Copy, Clone, PartialEq, Add, AddAssign, Serialize, Deserialize)]
pub struct RewardSummary(pub OnlineMeanVariance<f64>);

impl fmt::Display for RewardSummary {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Summary for RewardSummary {
    type Item = Reward;
    #[inline]
    fn push(&mut self, reward: &Reward) {
        self.0.push(reward.0)
    }
    #[inline]
    fn size(&self) -> u64 {
        self.0.count()
    }
    #[inline]
    fn merge(&mut self, other: Self) {
        self.0 += other.0
    }
}

impl Loggable for RewardSummary {
    fn log<L: StatsLogger + ?Sized>(
        &self,
        name: &'static str,
        logger: &mut L,
    ) -> Result<(), LogError> {
        self.0.log("reward", &mut logger.with_scope(name))
    }
}
