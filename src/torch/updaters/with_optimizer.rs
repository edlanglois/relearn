use super::super::{
    critic::Critic, history::PackedHistoryFeaturesView, optimizers::BuildOptimizer, policy::Policy,
};
use super::{
    BuildCriticUpdater, BuildPolicyUpdater, PolicyStats, UpdateCritic, UpdateCriticWithOptimizer,
    UpdatePolicy, UpdatePolicyWithOptimizer,
};
use crate::logging::TimeSeriesLogger;
use tch::nn::VarStore;

/// An updater constructed from an update rule and an optimizer.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct WithOptimizer<U, O> {
    pub update_rule: U,
    pub optimizer: O,
}

impl<U, O, AS> UpdatePolicy<AS> for WithOptimizer<U, O>
where
    U: UpdatePolicyWithOptimizer<O, AS>,
    AS: ?Sized,
{
    fn update_policy(
        &mut self,
        policy: &dyn Policy,
        critic: &dyn Critic,
        features: &dyn PackedHistoryFeaturesView,
        action_space: &AS,
        logger: &mut dyn TimeSeriesLogger,
    ) -> PolicyStats {
        self.update_rule.update_policy_with_optimizer(
            policy,
            critic,
            features,
            &mut self.optimizer,
            action_space,
            logger,
        )
    }
}

impl<U, O> UpdateCritic for WithOptimizer<U, O>
where
    U: UpdateCriticWithOptimizer<O>,
{
    fn update_critic(
        &mut self,
        critic: &dyn Critic,
        features: &dyn PackedHistoryFeaturesView,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        self.update_rule
            .update_critic_with_optimizer(critic, features, &mut self.optimizer, logger)
    }
}

impl<U, OC, AS> BuildPolicyUpdater<AS> for WithOptimizer<U, OC>
where
    U: UpdatePolicyWithOptimizer<OC::Optimizer, AS> + Clone,
    OC: BuildOptimizer,
    AS: ?Sized,
{
    type Updater = WithOptimizer<U, OC::Optimizer>;

    fn build_policy_updater(&self, vs: &VarStore) -> Self::Updater {
        WithOptimizer {
            update_rule: self.update_rule.clone(),
            optimizer: self.optimizer.build_optimizer(vs).unwrap(), // TODO: Error handling
        }
    }
}

impl<U, OC> BuildCriticUpdater for WithOptimizer<U, OC>
where
    U: UpdateCriticWithOptimizer<OC::Optimizer> + Clone,
    OC: BuildOptimizer,
{
    type Updater = WithOptimizer<U, OC::Optimizer>;

    fn build_critic_updater(&self, vs: &VarStore) -> Self::Updater {
        WithOptimizer {
            update_rule: self.update_rule.clone(),
            optimizer: self.optimizer.build_optimizer(vs).unwrap(), // TODO: Error handling
        }
    }
}
