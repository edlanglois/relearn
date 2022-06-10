//! Vanilla Policy Gradient (VPG)
use super::{
    Critic, HistoryFeatures, ParameterizedDistributionSpace, Policy, PolicyStats, PolicyUpdateRule,
    RuleOpt, RuleOptConfig, StatsLogger,
};
use crate::torch::optimizers::{opt_expect_ok_log, AdamConfig, Optimizer};
use crate::utils::distributions::ArrayDistribution;
use serde::{Deserialize, Serialize};
use tch::{COptimizer, Kind, Tensor};

/// Vanilla Policy Gradient (VPG) update rule.
///
/// Applies the REINFORCE algorithm with a [`Critic`] baseline.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VpgRule;

/// A [`LearningPolicy`][super::LearningPolicy] using [Vanilla Policy Gradient][VpgRule].
pub type Vpg<P, O = COptimizer> = RuleOpt<P, O, VpgRule>;
/// Configuration for [`Vpg`], a [Vanilla Policy Gradient][VpgRule] [`LearningPolicy`][1].
///
/// [1]: super::LearningPolicy
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
        features: &dyn HistoryFeatures,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) -> PolicyStats {
        let step_values = tch::no_grad(|| critic.step_values(features));

        let mut entropies = None;
        let mut policy_loss_fn = || {
            let action_dist_params = policy.seq_packed(features.observation_features());

            let action_distributions = action_space.distribution(action_dist_params.tensor());
            let log_probs = action_distributions.log_probs(features.actions().tensor());
            entropies.get_or_insert_with(|| action_distributions.entropy());
            -(log_probs * step_values.tensor()).mean(Kind::Float)
        };

        let result = optimizer.backward_step(&mut policy_loss_fn, logger);
        opt_expect_ok_log(result, "policy step error");

        let entropy = entropies.map(|e| e.mean(Kind::Float).into());
        PolicyStats { entropy }
    }
}
