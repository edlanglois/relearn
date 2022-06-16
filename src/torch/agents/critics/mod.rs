//! Critics for an actor-critic agent.
#![allow(clippy::use_self)] // false positive with serde derives
mod opt;
mod rtg;

pub use opt::{ValuesOpt, ValuesOptConfig};
pub use rtg::{RewardToGo, RewardToGoConfig};

use super::features::HistoryFeatures;
use crate::logging::StatsLogger;
use crate::torch::modules::SeqPacked;
use crate::torch::packed::PackedTensor;
use serde::{Deserialize, Serialize};
use tch::Device;

/// A critic for an [actor-critic agent][super::ActorCriticAgent].
///
/// Estimates the value (minus any per-state baseline) of each selected action in a collection of
/// experience. Learns from collected experience.
pub trait Critic {
    /// Value estimates of the selected actions offset by a baseline function of state.
    ///
    /// Formally, a estimate of `Q(a_t; o_0, ..., o_t) - b(o_0, ..., o_t)` for each time step `t`
    /// where `Q` is the discounted state-action value function and `b` is any baseline function
    /// that does not depend on the value of `a_t`. The environment is assumed to be partially
    /// observable so state is represented by the observation history.
    ///
    /// The returned values are suitable for use in REINFORCE policy gradient updates.
    ///
    /// # Design Note
    /// These "advantages" are more general than standard advantages, which require `b` to be the
    /// state value function. As far as I am aware, there is no name for "state-action value
    /// function with a state baseline" and the name "generalized advantages" is
    /// [taken][AdvantageFn::Gae]. "Advantages" was chosen as a reasonably evocative short name
    /// despite being technically incorrect.
    fn advantages(&self, features: &dyn HistoryFeatures) -> PackedTensor;

    /// Update the critic given a collection of experience features.
    fn update(&mut self, features: &dyn HistoryFeatures, logger: &mut dyn StatsLogger);
}

/// Build a [`Critic`].
pub trait BuildCritic {
    type Critic: Critic;

    fn build_critic(&self, in_dim: usize, discount_factor: f64, device: Device) -> Self::Critic;
}

/// Estimate baselined advantages from state value estimates and history features.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdvantageFn {
    /// Generalized Advantage Estimation
    ///
    /// # Reference
    /// "[High-Dimensional Continuous Control Using Generalized Advantage Estimation][gae]"
    /// by Schulman et al.
    ///
    /// The `gamma` parameter is implemented as [`ValuesOptConfig::max_discount_factor`] so that
    /// the learned value module uses the same discount factor.
    ///
    /// [gae]: https://arxiv.org/abs/1506.02438
    Gae {
        /// Shaping parameter in `[0, 1]` prioritizing sampled reward-to-go over the value module.
        ///
        /// Selects the degree to which advantage estimates rely on the value module estimates
        /// (low values) versus the empirical reward-to-go (high values).
        /// * `lambda = 0` is the 1-step temporal difference: `r_t +  γ * V(s_{t+1}) - V(s_t)`
        /// * `lambda = 1` is the reward-to-go with baseline: `sum_l(γ^l r_{t+l}) - V(s_t)`
        ///
        /// Lower values reduce variance but increase bias when the value function module is
        /// incorrect.
        lambda: f32,
    },
}

impl Default for AdvantageFn {
    fn default() -> Self {
        Self::Gae { lambda: 0.95 }
    }
}

impl AdvantageFn {
    /// Estimate baselined advantages of selected actions given a state value function module.
    pub fn advantages<M: SeqPacked + ?Sized>(
        &self,
        state_value_fn: &M,
        discount_factor: f32,
        features: &dyn HistoryFeatures,
    ) -> PackedTensor {
        match self {
            Self::Gae { lambda } => gae(state_value_fn, discount_factor, *lambda, features),
        }
    }
}

/// Discounted reward-to-go
///
/// # Args:
/// * `discount_factor` - Discount factor on future rewards. In `[0, 1]`.
/// * `features` - Experience features.
pub fn reward_to_go(discount_factor: f32, features: &dyn HistoryFeatures) -> PackedTensor {
    features
        .rewards()
        .discounted_cumsum_from_end(discount_factor)
}

/// Apply a state value function to `HistoryFeatures::extended_observations`.
///
/// # Args:
/// * `state_value_fn` - State value function estimator using past & present episode observations.
/// * `discount_factor` - Discount factor on future rewards. In `[0, 1]`.
/// * `features` - Experience features.
///
/// Returns a [`PackedTensor`] of sequences that are one longer than the episode lengths.
/// The final value is `0` if the episode ended and the next state value if interrupted.
pub fn eval_extended_state_values<M: SeqPacked + ?Sized>(
    state_value_fn: &M,
    features: &dyn HistoryFeatures,
) -> PackedTensor {
    let (extended_observation_features, is_invalid) = features.extended_observation_features();

    // Packed estimated values of the observed states
    let mut extended_estimated_values = state_value_fn
        .seq_packed(extended_observation_features)
        .batch_map(|t| t.squeeze_dim(-1));
    let _ = extended_estimated_values
        .tensor_mut()
        .masked_fill_(is_invalid.tensor(), 0.0);

    extended_estimated_values
}

/// One-step targets of a state value function.
///
/// # Args:
/// * `state_value_fn` - State value function estimator using past & present episode observations.
/// * `discount_factor` - Discount factor on future rewards. In `[0, 1]`.
/// * `features` - Experience features.
pub fn one_step_values<M: SeqPacked + ?Sized>(
    state_value_fn: &M,
    discount_factor: f32,
    features: &dyn HistoryFeatures,
) -> PackedTensor {
    // Estimated value for each of `step.next.into_inner().observation`
    let estimated_next_values =
        eval_extended_state_values(state_value_fn, features).view_trim_start(1);
    features
        .rewards()
        .batch_map_ref(|rewards| rewards + discount_factor * estimated_next_values.tensor())
}

/// One-step temporal difference residuals of a state value function
///
/// # Args:
/// * `state_value_fn` - State value function estimator using past & present episode observations.
/// * `discount_factor` - Discount factor on future rewards. In `[0, 1]`.
/// * `features` - Experience features.
pub fn temporal_differences<M: SeqPacked + ?Sized>(
    state_value_fn: &M,
    discount_factor: f32,
    features: &dyn HistoryFeatures,
) -> PackedTensor {
    let extended_state_values = eval_extended_state_values(state_value_fn, features);

    // Estimated values for each of `step.observation`
    let estimated_values = extended_state_values.trim_end(1);

    // Estimated value for each of `step.next.into_inner().observation`
    let estimated_next_values = extended_state_values.view_trim_start(1);

    features.rewards().batch_map_ref(|rewards| {
        rewards + discount_factor * estimated_next_values.tensor() - estimated_values.tensor()
    })
}

/// Generalized advantage estimation
///
/// # Reference
/// "[High-Dimensional Continuous Control Using Generalized Advantage Estimation][gae]"
/// by Schulman et al.
///
/// [gae]: https://arxiv.org/abs/1506.02438
///
/// # Args:
/// * `state_value_fn` - State value function estimator using past & present episode observations.
/// * `discount_factor` - Discount factor on future rewards. In `[0, 1]`.
/// * `lambda` - Parameter prioritizing sampled reward-to-go over the value module. In `[0, 1]`.
///     See [`AdvantageFn::Gae::lambda`].
/// * `features` - Experience features.
pub fn gae<M: SeqPacked + ?Sized>(
    state_value_fn: &M,
    discount_factor: f32,
    lambda: f32,
    features: &dyn HistoryFeatures,
) -> PackedTensor {
    let residuals = temporal_differences(state_value_fn, discount_factor, features);

    residuals.discounted_cumsum_from_end(lambda * discount_factor)
}

/// Target function for per-step selected-action value estimates.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum StepValueTarget {
    /// The empirical reward-to-go: discounted sum of future rewards.
    RewardToGo,
    /// One-step temporal-difference targets: `r_i + γ * V(s_{i+1})`
    OneStepTd,
}

impl Default for StepValueTarget {
    fn default() -> Self {
        Self::RewardToGo
    }
}

impl StepValueTarget {
    /// Generate state value targets for each state in a collection of experience.
    pub fn targets<M: SeqPacked + ?Sized>(
        &self,
        state_value_fn: &M,
        discount_factor: f32,
        features: &dyn HistoryFeatures,
    ) -> PackedTensor {
        match self {
            Self::RewardToGo => reward_to_go(discount_factor, features),
            Self::OneStepTd => one_step_values(state_value_fn, discount_factor, features),
        }
    }
}
