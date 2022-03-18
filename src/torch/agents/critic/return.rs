use super::super::learning_critic::{BuildLearningCritic, LearningCritic};
use super::{BuildCritic, Critic, Module, PackedHistoryFeaturesView};
use crate::logging::StatsLogger;
use serde::{Deserialize, Serialize};
use std::iter;
use tch::{Device, Tensor};

/// Value steps using the empirical discounted step return.
///
/// Also known as the Monte Carlo reward-to-go.
///
/// # Warning
/// Does not properly handle interrupted episodes.
/// This assumes that all episodes end with a reward of 0.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Return;

impl BuildCritic for Return {
    type Critic = Self;

    #[inline]
    fn build_critic(&self, _in_dim: usize, _: Device) -> Self::Critic {
        *self
    }
}

impl Module for Return {
    #[inline]
    fn shallow_clone(&self) -> Self
    where
        Self: Sized,
    {
        *self
    }

    #[inline]
    fn clone_to_device(&self, _: Device) -> Self
    where
        Self: Sized,
    {
        *self
    }

    #[inline]
    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(iter::empty())
    }

    #[inline]
    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(iter::empty())
    }
}

impl Critic for Return {
    #[inline]
    fn step_values(&self, features: &dyn PackedHistoryFeaturesView) -> Tensor {
        features.returns().shallow_clone()
    }

    #[inline]
    fn loss(&self, _: &dyn PackedHistoryFeaturesView) -> Option<Tensor> {
        None
    }
}

impl BuildLearningCritic for Return {
    type LearningCritic = Self;

    fn build_learning_critic(&self, _in_dim: usize, _: Device) -> Self::LearningCritic {
        *self
    }
}

/// Trivial "learning" --- nothing to learn
impl LearningCritic for Return {
    type Critic = Self;

    #[inline]
    fn critic_ref(&self) -> &Self::Critic {
        self
    }

    #[inline]
    fn into_critic(self) -> Self::Critic {
        self
    }

    #[inline]
    fn update_critic(&mut self, _: &dyn PackedHistoryFeaturesView, _: &mut dyn StatsLogger) {}
}
