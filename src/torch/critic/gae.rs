//! Generalized Advantage Estimation
use super::super::history::PackedHistoryFeaturesView;
use super::{Critic, CriticBuilder};
use crate::torch::{seq_modules::SequenceModule, ModuleBuilder};
use crate::utils::packed;
use tch::{nn::Path, Reduction, Tensor};

/// Generalized Advantage Estimator Config
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaeConfig<VB> {
    pub gamma: f64,
    pub lambda: f64,
    pub value_fn_config: VB,
}

impl<VB: Default> Default for GaeConfig<VB> {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            value_fn_config: Default::default(),
        }
    }
}

impl<VB, V> CriticBuilder<Gae<V>> for GaeConfig<VB>
where
    VB: ModuleBuilder<V>,
    V: SequenceModule,
{
    fn build_critic(&self, vs: &Path, in_dim: usize) -> Gae<V> {
        Gae {
            gamma: self.gamma,
            lambda: self.lambda,
            value_fn: self.value_fn_config.build_module(vs, in_dim, 1),
        }
    }
}

/// Generalized Advantage Estimator critic.
///
/// # Note
/// Currently does not properly handle non-terminal end-of-episode.
/// This assumes that all episodes end with a reward of `0`.
///
/// # Reference
/// High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR  2016
/// by John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel
/// <https://arxiv.org/pdf/1506.02438.pdf>
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gae<V> {
    /// Clips the environment discount factor to be no more than this.
    pub gamma: f64,

    /// Advantage interpolation factor between one-step residuals (=0) and full return (=1).
    pub lambda: f64,

    /// State value function module.
    pub value_fn: V,
}

impl<V> Critic for Gae<V>
where
    V: SequenceModule,
{
    fn trainable(&self) -> bool {
        true
    }

    fn discount_factor(&self, env_discount_factor: f64) -> f64 {
        env_discount_factor.min(self.gamma)
    }

    fn seq_packed(&self, features: &dyn PackedHistoryFeaturesView) -> Tensor {
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
}
