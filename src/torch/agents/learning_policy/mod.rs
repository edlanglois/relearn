//! Learning policies
mod ppo;
mod trpo;
mod vpg;

pub use ppo::{Ppo, PpoConfig, PpoRule};
pub use trpo::{Trpo, TrpoConfig, TrpoRule};
pub use vpg::{Vpg, VpgConfig, VpgRule};

use super::critic::Critic;
use super::features::HistoryFeatures;
use super::policy::{BuildPolicy, Policy};
use super::{RuleOpt, RuleOptConfig};
use crate::logging::StatsLogger;
use crate::spaces::ParameterizedDistributionSpace;
use crate::torch::modules::Module;
use crate::torch::optimizers::BuildOptimizer;
use serde::{Deserialize, Serialize};
use tch::{Device, Tensor};

/// A [`Policy`] that can learn from collected history features and a critic.
pub trait LearningPolicy {
    /// Type of the internal stored policy.
    type Policy: Policy + Module;

    /// Cheap reference to the internal policy.
    fn policy_ref(&self) -> &Self::Policy;

    /// Convert into the inernal policy.
    fn into_policy(self) -> Self::Policy;

    /// Update the internal policy.
    ///
    /// # Args
    /// * `critic`   - A [`Critic`] assigning values to the steps in `features`.
    /// * `features` - Packed history features collected according to the current policy.
    /// * `action_space` - The environment action space.
    /// * `logger`   - A logger for update statistics.
    fn update_policy<AS: ParameterizedDistributionSpace<Tensor>>(
        &mut self,
        critic: &dyn Critic,
        features: &dyn HistoryFeatures,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) -> PolicyStats;
}

/// Build a [`LearningPolicy`].
pub trait BuildLearningPolicy {
    type LearningPolicy: LearningPolicy;

    /// Build a learning policy.
    ///
    /// # Args
    /// * `in_dim`  - Policy input dimension (number of observation features).
    /// * `out_dim` - Policy output dimension (number of action distribution parameters).
    /// * `device`  - Device on which the policy tensors are placed.
    fn build_learning_policy(
        &self,
        in_dim: usize,
        out_dim: usize,
        device: Device,
    ) -> Self::LearningPolicy;
}

/// Common summary statistics from a policy update.
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyStats {
    /// Policy entropy (unspecified whether before, after, or during update).
    pub entropy: Option<f64>,
}

/// Update an external policy given an optimizer for that policy's variables.
pub trait PolicyUpdateRule<P, O> {
    fn update_external_policy<AS: ParameterizedDistributionSpace<Tensor>>(
        &self,
        policy: &P,
        optimizer: &mut O,
        critic: &dyn Critic,
        features: &dyn HistoryFeatures,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) -> PolicyStats;
}

impl<P, O, U> LearningPolicy for RuleOpt<P, O, U>
where
    P: Policy + Module,
    U: PolicyUpdateRule<P, O>,
{
    type Policy = P;
    #[inline]
    fn policy_ref(&self) -> &Self::Policy {
        &self.module
    }
    #[inline]
    fn into_policy(self) -> Self::Policy {
        self.module
    }
    #[inline]
    fn update_policy<AS: ParameterizedDistributionSpace<Tensor>>(
        &mut self,
        critic: &dyn Critic,
        features: &dyn HistoryFeatures,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) -> PolicyStats {
        self.update_rule.update_external_policy(
            &self.module,
            &mut self.optimizer,
            critic,
            features,
            action_space,
            logger,
        )
    }
}

impl<PB, OB, U> BuildLearningPolicy for RuleOptConfig<PB, OB, U>
where
    PB: BuildPolicy,
    PB::Policy: Module,
    OB: BuildOptimizer,
    U: PolicyUpdateRule<PB::Policy, OB::Optimizer> + Clone,
{
    type LearningPolicy = RuleOpt<PB::Policy, OB::Optimizer, U>;

    #[inline]
    fn build_learning_policy(
        &self,
        in_dim: usize,
        out_dim: usize,
        device: Device,
    ) -> Self::LearningPolicy {
        let module = self.module_config.build_policy(in_dim, out_dim, device);
        RuleOpt::new(
            module,
            &self.optimizer_config,
            self.update_rule_config.clone(),
        )
    }
}
