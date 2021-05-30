//! RL step value functions
mod gae;

pub use gae::{Gae, GaeConfig};

use super::history::PackedHistoryFeaturesView;
use tch::{nn::Path, Tensor};

/// A step value function for use in the policy gradient.
pub trait StepValue {
    /// Whether this step values function has trainable internal parameters
    fn trainable(&self) -> bool;

    /// The discount factor to use when calculating step returns.
    ///
    /// # Args
    /// * `env_discount_factor` - The discount factor specified by the environment.
    fn discount_factor(&self, env_discount_factor: f64) -> f64 {
        env_discount_factor
    }

    /// Evaluate the step values packed sequences of steps.
    ///
    /// # Args
    /// * `features` - A view of the packed step history features.
    ///
    /// # Return
    /// Packed step values. A 1D f32 tensor with the same shape as `rewards` and `returns`.
    fn seq_packed(&self, features: &dyn PackedHistoryFeaturesView) -> Tensor;

    /// The loss of any trainable internal variables given the observed history features.
    ///
    /// Returns None if and only if trainable() is false.
    fn loss(&self, features: &dyn PackedHistoryFeaturesView) -> Option<Tensor>;
}

/// Build a [`StepValue`] instance.
pub trait StepValueBuilder<T> {
    /// Build a new [`StepValue`] instance.
    ///
    /// # Args
    /// * `vs` - Variable store and namespace.
    /// * `in_dim` - Number of input feature dimensions.
    fn build_step_value(&self, vs: &Path, in_dim: usize) -> T;
}

/// Value steps using the empirical discounted step return.
///
/// Also known as the Monte Carlo reward-to-go.
///
/// # Note
/// Currently does not properly handle non-terminal end-of-episode.
/// This assumes that all episodes end with a reward of 0.
#[derive(Debug)]
pub struct Return;

impl Default for Return {
    fn default() -> Self {
        Self
    }
}

impl StepValue for Return {
    fn trainable(&self) -> bool {
        false
    }

    fn seq_packed(&self, features: &dyn PackedHistoryFeaturesView) -> Tensor {
        features.returns().shallow_clone()
    }

    fn loss(&self, _features: &dyn PackedHistoryFeaturesView) -> Option<Tensor> {
        None
    }
}

impl StepValueBuilder<Self> for Return {
    fn build_step_value(&self, _: &Path, _: usize) -> Self {
        Return
    }
}

impl<T: StepValue + ?Sized> StepValue for Box<T> {
    fn trainable(&self) -> bool {
        T::trainable(self)
    }

    fn discount_factor(&self, env_discount_factor: f64) -> f64 {
        T::discount_factor(self, env_discount_factor)
    }

    fn seq_packed(&self, features: &dyn PackedHistoryFeaturesView) -> Tensor {
        T::seq_packed(self, features)
    }

    fn loss(&self, features: &dyn PackedHistoryFeaturesView) -> Option<Tensor> {
        T::loss(self, features)
    }
}
