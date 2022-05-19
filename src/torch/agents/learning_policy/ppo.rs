//! Proximal Policy Optimization (PPO)
use super::{
    Critic, HistoryFeatures, ParameterizedDistributionSpace, Policy, PolicyStats, PolicyUpdateRule,
    RuleOpt, RuleOptConfig, StatsLogger,
};
use crate::torch::optimizers::{AdamConfig, Optimizer};
use crate::utils::distributions::ArrayDistribution;
use serde::{Deserialize, Serialize};
use tch::{COptimizer, Kind, Tensor};

/// Proximal Policy Optimization (PPO) with a clipped objective.
///
/// # Reference
/// [Proximal Policy Optimization Algorithms][ppo] by Schulman et al.
///
/// [ppo]: https://arxiv.org/abs/1707.06347
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct PpoRule {
    pub num_epochs: u64,
    // TODO: Support minibatches
    // pub minibatch_size: usize,
    pub clip_distance: f64,
}

impl Default for PpoRule {
    #[inline]
    fn default() -> Self {
        Self {
            num_epochs: 10,
            clip_distance: 0.2,
        }
    }
}

/// A [`LearningPolicy`][super::LearningPolicy] using [Proximal Policy Optimization][PpoRule].
pub type Ppo<P, O = COptimizer> = RuleOpt<P, O, PpoRule>;
/// Configuration for [`Ppo`], a [Proximal Policy Optimization][PpoRule] [`LearningPolicy`][1].
///
/// [1]: super::LearningPolicy
pub type PpoConfig<PB, OB = AdamConfig> = RuleOptConfig<PB, OB, PpoRule>;

impl<P, O> PolicyUpdateRule<P, O> for PpoRule
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
        let observation_features = features.observation_features();
        let actions = features.actions().tensor();

        let (step_values, initial_log_probs, initial_policy_entropy) = {
            let _no_grad = tch::no_grad_guard();

            let step_values = critic.step_values(features);
            let policy_output = policy.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());
            let log_probs = distribution.log_probs(actions);
            let entropy = distribution.entropy().mean(Kind::Float);

            (step_values, log_probs, f64::from(entropy))
        };

        let policy_surrogate_loss_fn = || {
            let policy_output = policy.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());
            let log_probs = distribution.log_probs(actions);

            let likelihood_ratio = (log_probs - &initial_log_probs).exp();
            let clipped_likelihood_ratio =
                likelihood_ratio.clip(1.0 - self.clip_distance, 1.0 + self.clip_distance);

            (likelihood_ratio * step_values.tensor())
                .min_other(&(clipped_likelihood_ratio * step_values.tensor()))
                .mean(Kind::Float)
                .neg()
        };

        let _ = optimizer
            .backward_step(&policy_surrogate_loss_fn, logger)
            .unwrap();

        PolicyStats {
            entropy: Some(initial_policy_entropy),
        }
    }
}
