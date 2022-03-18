//! Learning critics
mod gradient;

pub use gradient::{GradOpt, GradOptConfig, GradOptRule};

use super::critic::{BuildCritic, Critic};
use super::features::PackedHistoryFeaturesView;
use super::{RuleOpt, RuleOptConfig};
use crate::logging::StatsLogger;
use crate::torch::optimizers::BuildOptimizer;
use tch::Device;

/// A [`Critic`] that can learn from collected history features.
pub trait LearningCritic {
    /// Type of the internal stored critic.
    type Critic: Critic;

    /// Cheap reference to the internal critic.
    fn critic_ref(&self) -> &Self::Critic;

    /// Convert into the internal critic.
    fn into_critic(self) -> Self::Critic;

    /// Update the internal critic.
    ///
    /// # Args
    /// * `features` - Packed history features.
    /// * `logger`   - A logger for update statistics.
    fn update_critic(
        &mut self,
        features: &dyn PackedHistoryFeaturesView,
        logger: &mut dyn StatsLogger,
    );
}

/// Build a [`LearningCritic`].
pub trait BuildLearningCritic {
    type LearningCritic: LearningCritic;

    /// Build a learning critic.
    ///
    /// # Args
    /// * `in_dim` - Critic input dimension (number of observation features).
    /// * `device` - Device on which the critic tensors are placed.
    fn build_learning_critic(&self, in_dim: usize, device: Device) -> Self::LearningCritic;
}

/// Update an external critic given an optimizer for that critic's variables.
pub trait CriticUpdateRule<C, O> {
    fn update_external_critic(
        &mut self,
        critic: &C,
        optimizer: &mut O,
        features: &dyn PackedHistoryFeaturesView,
        logger: &mut dyn StatsLogger,
    );
}

impl<C, O, U> LearningCritic for RuleOpt<C, O, U>
where
    C: Critic,
    U: CriticUpdateRule<C, O>,
{
    type Critic = C;
    #[inline]
    fn critic_ref(&self) -> &Self::Critic {
        &self.module
    }
    #[inline]
    fn into_critic(self) -> Self::Critic {
        self.module
    }
    #[inline]
    fn update_critic(
        &mut self,
        features: &dyn PackedHistoryFeaturesView,
        logger: &mut dyn StatsLogger,
    ) {
        self.update_rule
            .update_external_critic(&self.module, &mut self.optimizer, features, logger)
    }
}

impl<CB, OB, U> BuildLearningCritic for RuleOptConfig<CB, OB, U>
where
    CB: BuildCritic,
    OB: BuildOptimizer,
    U: CriticUpdateRule<CB::Critic, OB::Optimizer> + Clone,
{
    type LearningCritic = RuleOpt<CB::Critic, OB::Optimizer, U>;

    #[inline]
    fn build_learning_critic(&self, in_dim: usize, device: Device) -> Self::LearningCritic {
        let module = self.module_config.build_critic(in_dim, device);
        RuleOpt::new(
            module,
            &self.optimizer_config,
            self.update_rule_config.clone(),
        )
    }
}
