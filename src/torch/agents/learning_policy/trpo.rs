//! Trust Region Policy Optimization (TRPO)
use super::{
    Critic, HistoryFeatures, ParameterizedDistributionSpace, Policy, PolicyStats, PolicyUpdateRule,
    RuleOpt, RuleOptConfig, StatsLogger,
};
use crate::torch::{
    backends::WithCudnnEnabled,
    optimizers::{
        ConjugateGradientOptimizer, ConjugateGradientOptimizerConfig, OptimizerStepError,
        TrustRegionOptimizer,
    },
};
use crate::utils::distributions::ArrayDistribution;
use log::warn;
use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

/// Trust Region Policy Optimization (PPO) with a clipped objective.
///
/// # Reference
/// [Trust Region Policy Optimization][trpo] by Schulman et al.
///
/// [trpo]: https://arxiv.org/abs/1502.05477
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrpoRule {
    /// Maximum policy KL divergence when taking a step.
    pub max_policy_step_kl: f64,
}

impl Default for TrpoRule {
    #[inline]
    fn default() -> Self {
        Self {
            // This step size was used by all experiments in Schulman's TRPO paper.
            max_policy_step_kl: 0.01,
        }
    }
}

/// A [`LearningPolicy`][super::LearningPolicy] using [Trust Region Policy Optimization][TrpoRule].
pub type Trpo<P, O = ConjugateGradientOptimizer> = RuleOpt<P, O, TrpoRule>;
/// Configuration for [`Trpo`], a [Trust Region Policy Optimization][TrpoRule] [`LearningPolicy`][1].
///
/// [1]: super::LearningPolicy
pub type TrpoConfig<PB, OB = ConjugateGradientOptimizerConfig> = RuleOptConfig<PB, OB, TrpoRule>;

impl<P, O> PolicyUpdateRule<P, O> for TrpoRule
where
    P: Policy,
    O: TrustRegionOptimizer,
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
        let _cudnn_disable_guard = if policy.has_cudnn_second_derivatives() {
            None
        } else {
            Some(WithCudnnEnabled::new(false))
        };

        let observation_features = features.observation_features();
        let actions = features.actions().tensor();

        let (step_values, initial_distribution, initial_log_probs, initial_policy_entropy) = {
            let _no_grad = tch::no_grad_guard();

            let step_values = critic.step_values(features);
            let policy_output = policy.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());
            let log_probs = distribution.log_probs(actions);
            let entropy = distribution.entropy().mean(Kind::Float);

            (step_values, distribution, log_probs, f64::from(entropy))
        };

        let policy_loss_distance_fn = || {
            let policy_output = policy.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());

            let log_probs = distribution.log_probs(actions);
            let likelihood_ratio = (log_probs - &initial_log_probs).exp();
            let loss = -(likelihood_ratio * step_values.tensor()).mean(Kind::Float);

            // NOTE:
            // The [TRPO paper] and [Garage] use `KL(old_policy || new_policy)` while
            // [Spinning Up] uses `KL(new_policy || old_policy)`.
            //
            // I do not know why Spinning Up differs. I follow the TRPO paper and Garage.
            //
            // [TRPO paper]: <https://arxiv.org/abs/1502.05477>
            // [Garage]: <https://garage.readthedocs.io/en/latest/user/algo_trpo.html>
            // [Spinning Up]: <https://spinningup.openai.com/en/latest/algorithms/trpo.html>
            let distance = initial_distribution
                .kl_divergence_from(&distribution)
                .mean(Kind::Float);

            (loss, distance)
        };

        let result = optimizer.trust_region_backward_step(
            &policy_loss_distance_fn,
            self.max_policy_step_kl,
            logger,
        );

        if let Err(error) = result {
            match error {
                OptimizerStepError::NaNLoss => panic!("NaN loss in policy optimization"),
                OptimizerStepError::NaNConstraint => {
                    panic!("NaN constraint in policy optimization")
                }
                err => warn!("error in policy step: {}", err),
            };
        }

        PolicyStats {
            entropy: Some(initial_policy_entropy),
        }
    }
}
