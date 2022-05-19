use super::super::learning_critic::{BuildLearningCritic, LearningCritic};
use super::{BuildCritic, Critic, Module, PackedHistoryFeaturesView};
use crate::logging::StatsLogger;
use crate::torch::packed::PackedTensor;
use serde::{Deserialize, Serialize};
use std::iter;
use tch::{Device, Tensor};

/// Value steps using the empirical discounted step return.
///
/// Also known as the Monte Carlo reward-to-go.
///
/// # Warning
/// This assumes that all episodes end with a reward of 0 (including interrupted episodes).
/// There is no alternative since this critic lacks an estimated value model.
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
    #[allow(clippy::cast_possible_truncation)]
    fn step_values(&self, features: &dyn PackedHistoryFeaturesView) -> PackedTensor {
        // Note: This assumes that all episodes end with 0 return.
        features
            .rewards()
            .discounted_cumsum_from_end(features.discount_factor() as f32)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{BooleanSpace, IndexSpace};
    use crate::torch::agents::features::tests::{history, StoredHistory};
    use rstest::rstest;

    #[rstest]
    #[allow(clippy::needless_pass_by_value)]
    fn step_values(history: StoredHistory<BooleanSpace, IndexSpace>) {
        let features = history.features();
        let actual = Return.step_values(&features);
        let expected = &Tensor::of_slice(&[
            -0.65341, 3.439, 5.42, 3.0, //
            0.3851, 2.71, 3.8, //
            1.539, 1.9, 2.0, //
            1.71, 1.0, //
            1.9, //
            1.0f32,
        ]);
        assert!(expected.allclose(actual.tensor(), 1e-5, 1e-5, false));
    }
}
