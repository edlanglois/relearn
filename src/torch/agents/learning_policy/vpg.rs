//! Vanilla Policy Gradient (VPG)
use super::{
    Critic, PackedHistoryFeaturesView, ParameterizedDistributionSpace, Policy, PolicyStats,
    PolicyUpdateRule, RuleOpt, RuleOptConfig, StatsLogger,
};
use crate::torch::optimizers::{AdamConfig, Optimizer};
use crate::utils::distributions::ArrayDistribution;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use tch::{COptimizer, Kind, Tensor};

/// Vanilla Policy Gradient (VPG) update rule.
///
/// Applies the REINFORCE algorithm with a [`Critic`] baseline.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VpgRule;

/// A [`LearningPolicy`] using [Vanilla Policy Gradient][VpgRule].
pub type Vpg<P, O = COptimizer> = RuleOpt<P, O, VpgRule>;
/// Configuration for [`Vpg`], a [Vanilla Policy Gradient][VpgRule] [`LearningPolicy`].
pub type VpgConfig<PB, OB = AdamConfig> = RuleOptConfig<PB, OB, VpgRule>;

impl<P, O> PolicyUpdateRule<P, O> for VpgRule
where
    P: Policy,
    O: Optimizer,
{
    fn update_external_policy<AS: ParameterizedDistributionSpace<Tensor>>(
        &self,
        policy: &P,
        optimizer: &mut O,
        critic: &dyn Critic,
        features: &dyn PackedHistoryFeaturesView,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) -> PolicyStats {
        let step_values = tch::no_grad(|| critic.step_values(features));

        let entropies = Cell::new(None);
        let policy_loss_fn = || {
            let action_dist_params = policy.seq_packed(
                features.observation_features(),
                features.batch_sizes_tensor(),
            );

            let action_distributions = action_space.distribution(&action_dist_params);
            let log_probs = action_distributions.log_probs(features.actions());
            entropies.set(Some(action_distributions.entropy()));
            -(log_probs * &step_values).mean(Kind::Float)
        };

        let _ = optimizer.backward_step(&policy_loss_fn, logger).unwrap();

        let entropy = entropies.into_inner().map(|e| e.mean(Kind::Float).into());
        PolicyStats { entropy }
    }
}
