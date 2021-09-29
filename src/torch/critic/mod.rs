//! RL Critics
mod gae;

pub use gae::{Gae, GaeConfig};

use super::history::PackedHistoryFeaturesView;
use tch::{nn::Path, Tensor};

/// Critic for a reinforcement learning environment.
///
/// Assigns a score to each state-action pair in a trajectory.
/// The score may depend on past states and actions in the trajectory and on future rewards.
/// Higher scores represent better outcomes than lower scores.
///
/// This use of "Critic" is prehaps more expansive than the typical use:
/// it does not just refer to a runtime evaluator of expected future reward given the observed
/// trajectory so far, but instead includes a retrospective evaluation of states and actions given
/// the empirical future trajectory.
pub trait Critic {
    /// Whether this critic has trainable internal parameters
    fn trainable(&self) -> bool;

    /// Get the discount factor to use when calculating step returns.
    ///
    /// Some critics will use a reduced discount factor for reduced variance
    /// at the cost of extra bias.
    ///
    /// This information is required by the caller so that it can be provided to the
    /// `PackedHistoryFeaturesView`, which only supports a single choice of discount factor.
    ///
    /// # Args
    /// * `env_discount_factor` - The discount factor specified by the environment.
    fn discount_factor(&self, env_discount_factor: f64) -> f64 {
        env_discount_factor
    }

    /// Provide values for a packed sequence of steps.
    ///
    /// # Args
    /// * `features` - A view of the packed step history features.
    ///
    /// # Return
    /// Packed step values. A 1D f32 tensor with the same shape as `rewards` and `returns`.
    fn seq_packed(&self, features: &dyn PackedHistoryFeaturesView) -> Tensor;

    /// The loss of any trainable internal variables given the observed history features.
    ///
    /// Returns `None` if and only if [`Critic::trainable`] is false.
    fn loss(&self, features: &dyn PackedHistoryFeaturesView) -> Option<Tensor>;
}

/// Build a [`Critic`] instance.
pub trait BuildCritic {
    type Critic;

    /// Build a new [`Critic`] instance.
    ///
    /// # Args
    /// * `vs` - Variable store and namespace.
    /// * `in_dim` - Number of input feature dimensions.
    fn build_critic(&self, vs: &Path, in_dim: usize) -> Self::Critic;
}

/// Value steps using the empirical discounted step return.
///
/// Also known as the Monte Carlo reward-to-go.
///
/// # Note
/// Currently does not properly handle non-terminal end-of-episode.
/// This assumes that all episodes end with a reward of 0.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Return;

impl Critic for Return {
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

impl BuildCritic for Return {
    type Critic = Self;

    fn build_critic(&self, _vs: &Path, _in_dim: usize) -> Self::Critic {
        Self
    }
}

impl<T: Critic + ?Sized> Critic for Box<T> {
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
