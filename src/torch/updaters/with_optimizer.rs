use super::super::history::PackedHistoryFeaturesView;
use super::super::optimizers::OptimizerBuilder;
use super::{
    PolicyStats, UpdateCritic, UpdateCriticWithOptimizer, UpdatePolicy, UpdatePolicyWithOptimizer,
    UpdaterBuilder,
};
use crate::logging::TimeSeriesLogger;
use tch::nn::VarStore;

/// An updater constructed from an update rule and an optimizer.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct WithOptimizer<U, O> {
    pub update_rule: U,
    pub optimizer: O,
}

impl<U, P, C, O, AS> UpdatePolicy<P, C, AS> for WithOptimizer<U, O>
where
    U: UpdatePolicyWithOptimizer<P, C, O, AS>,
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

impl<U, C, O> UpdateCritic<C> for WithOptimizer<U, O>
where
    U: UpdateCriticWithOptimizer<C, O>,
    C: ?Sized,
{
    fn update_critic(
        &mut self,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        self.update_rule
            .update_critic_with_optimizer(critic, features, &mut self.optimizer, logger)
    }
}

impl<U, O, OB> UpdaterBuilder<WithOptimizer<U, O>> for WithOptimizer<U, OB>
where
    U: Clone,
    OB: OptimizerBuilder<O>,
{
    fn build_updater(&self, vs: &VarStore) -> WithOptimizer<U, O> {
        WithOptimizer {
            update_rule: self.update_rule.clone(),
            optimizer: self.optimizer.build_optimizer(vs).unwrap(),
        }
    }
}
