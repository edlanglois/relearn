use super::{BuildCritic, Critic, Module, PackedHistoryFeaturesView};
use crate::torch::modules::{BuildModule, SequenceModule};
use crate::utils::packed;
use serde::{Deserialize, Serialize};
use tch::{Device, Reduction, Tensor};

/// Configuration for the Generalized Advantage Estimator critic ([`Gae`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GaeConfig<VC> {
    pub gamma: f64,
    pub lambda: f64,
    pub value_fn_config: VC,
}

impl<VC: Default> Default for GaeConfig<VC> {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            value_fn_config: Default::default(),
        }
    }
}

impl<VC> BuildCritic for GaeConfig<VC>
where
    VC: BuildModule,
    VC::Module: SequenceModule,
{
    type Critic = Gae<VC::Module>;

    fn build_critic(&self, in_dim: usize, device: Device) -> Self::Critic {
        Gae {
            gamma: self.gamma,
            lambda: self.lambda,
            value_fn: self.value_fn_config.build_module(in_dim, 1, device),
        }
    }
}

/// Generalized Advantage Estimator (GAE) [critic][Critic].
///
/// # Warning
/// Does not properly handle interrupted episodes.
/// This assumes that all episodes end with a reward of 0.
///
/// # Reference
/// High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR  2016
/// by John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel
/// <https://arxiv.org/pdf/1506.02438.pdf>
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gae<V> {
    /// Clips the environment discount factor to be no more than this.
    pub gamma: f64,

    /// Advantage interpolation factor between one-step residuals (=0) and full return (=1).
    pub lambda: f64,

    /// State value function module.
    pub value_fn: V,
}

impl<V: Module> Module for Gae<V> {
    #[inline]
    fn shallow_clone(&self) -> Self
    where
        Self: Sized,
    {
        Self {
            gamma: self.gamma,
            lambda: self.lambda,
            value_fn: self.value_fn.shallow_clone(),
        }
    }

    #[inline]
    fn clone_to_device(&self, device: Device) -> Self
    where
        Self: Sized,
    {
        Self {
            gamma: self.gamma,
            lambda: self.lambda,
            value_fn: self.value_fn.clone_to_device(device),
        }
    }

    #[inline]
    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        self.value_fn.variables()
    }

    #[inline]
    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        self.value_fn.trainable_variables()
    }

    #[inline]
    fn has_cudnn_second_derivatives(&self) -> bool {
        self.value_fn.has_cudnn_second_derivatives()
    }
}

impl<V> Critic for Gae<V>
where
    V: SequenceModule,
{
    fn step_values(&self, features: &dyn PackedHistoryFeaturesView) -> Tensor {
        // Packed estimated values of the observed states
        let estimated_values = self
            .value_fn
            .seq_packed(
                features.observation_features(),
                features.batch_sizes_tensor(),
            )
            .squeeze_dim(-1);

        // Packed estimated value for the observed successor states.
        // Assumes that all end-of-episodes are terminal and have value 0.
        //
        // More generally, we should apply the value function to last_step.next_observation.
        // But this is tricky since the value function can be a sequential module and require the
        // state from the rest of the episode.
        let estimated_next_values =
            packed::packed_tensor_push_shift(&estimated_values, features.batch_sizes(), 0.0);

        let discount_factor = features.discount_factor();

        // Packed one-step TD residuals.
        let residuals =
            features.rewards() + discount_factor * estimated_next_values - estimated_values;

        // Packed step action advantages
        let advantages = packed::packed_tensor_discounted_cumsum_from_end(
            &residuals,
            features.batch_sizes(),
            self.lambda * discount_factor,
        );

        advantages
    }

    fn loss(&self, features: &dyn PackedHistoryFeaturesView) -> Option<Tensor> {
        Some(
            self.value_fn
                .seq_packed(
                    features.observation_features(),
                    features.batch_sizes_tensor(),
                )
                .squeeze_dim(-1)
                .mse_loss(features.returns(), Reduction::Mean),
        )
    }

    #[inline]
    fn discount_factor(&self, env_discount_factor: f64) -> f64 {
        env_discount_factor.min(self.gamma)
    }
}
