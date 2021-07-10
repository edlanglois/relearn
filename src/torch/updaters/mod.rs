//! Updaters for RL agent components
//!
//! # Terminology
//! * _Updater_ ([`UpdatePolicy`] and [`UpdateCritic`])
//!     - Self-contained update of a policy or critic.
//! * _Update Rule_ ([`UpdatePolicyWithOptimizer`] and [`UpdateCriticWithOptimizer`])
//!     - Update a policy or critic with an external optimizer.
//!
//! These are distinct because it is simpler to implement an update rule but using an updater
//! allows details about the optimizer type to be hidden.
mod critic_loss;
mod policy_gradient;
mod trpo;
mod with_optimizer;

pub use critic_loss::CriticLossUpdateRule;
pub use policy_gradient::PolicyGradientUpdateRule;
pub use trpo::TrpoPolicyUpdateRule;
pub use with_optimizer::WithOptimizer;

use super::history::PackedHistoryFeaturesView;
use crate::logging::TimeSeriesLogger;
use tch::nn::VarStore;

// TODO: Remove ActionSpace

/// Build an updater
pub trait UpdaterBuilder<U> {
    /// Build an updater for the trainable variables in a variable store.
    fn build_updater(&self, vs: &VarStore) -> U;
}

/// Self-contained policy updater.
pub trait UpdatePolicy<P, C, AS>
where
    P: ?Sized,
    C: ?Sized,
    AS: ?Sized,
{
    /// Update policy variables (manged internally).
    ///
    /// # Args
    /// * `policy` - Policy module to update.
    ///     The set of variables to update is stored internally by the updater, not obtained from
    ///     the policy, so only the policy for which the updater was initialized should be used.
    /// * `critic` - A critic assigning values to the steps of `features`.
    /// * `features` - Packed history features collected with the current policy variable values.
    /// * `action_space` - The environment action space.
    /// * `logger` - A logger for update statistics.
    fn update_policy(
        &mut self,
        policy: &P,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        action_space: &AS,
        logger: &mut dyn TimeSeriesLogger,
    ) -> PolicyStats;
}

impl<P, C, AS> UpdatePolicy<P, C, AS> for Box<dyn UpdatePolicy<P, C, AS>>
where
    P: ?Sized,
    C: ?Sized,
    AS: ?Sized,
{
    fn update_policy(
        &mut self,
        policy: &P,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        action_space: &AS,
        logger: &mut dyn TimeSeriesLogger,
    ) -> PolicyStats {
        self.as_mut()
            .update_policy(policy, critic, features, action_space, logger)
    }
}

/// A policy update rule using an external optimizer.
pub trait UpdatePolicyWithOptimizer<P, C, O, AS>
where
    P: ?Sized,
    C: ?Sized,
    O: ?Sized,
    AS: ?Sized,
{
    /// Update policy variables using an external optimizer.
    ///
    /// # Args
    /// * `policy` - Policy module to update. The variables to update are managed by `optimizer`.
    /// * `critic` - A critic assigning values to the steps of `features`.
    /// * `features` - Packed history features collected with the current policy variable values.
    /// * `optimizer` - An optimizer on the policy variables.
    /// * `action_space` - The environment action space.
    /// * `logger` - A logger for update statistics.
    fn update_policy_with_optimizer(
        &self,
        policy: &P,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        optimizer: &mut O,
        action_space: &AS,
        logger: &mut dyn TimeSeriesLogger,
    ) -> PolicyStats;
}

/// Common statistics from a policy update.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct PolicyStats {
    pub entropy: Option<f64>,
}

/// Self-contained critic updater.
pub trait UpdateCritic<C>
where
    C: ?Sized,
{
    /// Update critic variables (manged internally).
    ///
    /// # Args
    /// * `critic` - The critic to update.
    ///     The set of variables to update is stored internally by the updater, not obtained from
    ///     the policy, so only the policy for which the updater was initialized should be used.
    /// * `features` - Packed history features collected with the current policy variable values.
    /// * `logger` - A logger for update statistics.
    fn update_critic(
        &mut self,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        logger: &mut dyn TimeSeriesLogger,
    );
}

impl<C: ?Sized> UpdateCritic<C> for Box<dyn UpdateCritic<C>> {
    fn update_critic(
        &mut self,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        self.as_mut().update_critic(critic, features, logger)
    }
}

/// A critic update rule using an external optimizer.
pub trait UpdateCriticWithOptimizer<C, O>
where
    C: ?Sized,
    O: ?Sized,
{
    /// Update critic variables using an external optimizer.
    ///
    /// # Args
    /// * `critic` - The critic to update.
    ///     The set of variables to update is stored internally by the updater, not obtained from
    ///     the policy, so only the policy for which the updater was initialized should be used.
    /// * `features` - Packed history features collected with the current policy variable values.
    /// * `optimizer` - An optimizer on the policy variables.
    /// * `logger` - A logger for update statistics.
    fn update_critic_with_optimizer(
        &self,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        optimizer: &mut O,
        logger: &mut dyn TimeSeriesLogger,
    );
}
