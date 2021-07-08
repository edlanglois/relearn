use super::super::{
    backends::{CudnnSupport, WithCudnnEnabled},
    critic::Critic,
    history::PackedHistoryFeaturesView,
    optimizers::{OptimizerStepError, TrustRegionOptimizer},
    seq_modules::SequenceModule,
};
use super::{PolicyStats, UpdatePolicyWithOptimizer};
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::ParameterizedDistributionSpace;
use crate::utils::distributions::ArrayDistribution;
use tch::{Kind, Tensor};

/// Trust region policy update rule.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TrpoPolicyUpdateRule {
    /// Maximum policy KL divergence when taking a step.
    pub max_policy_step_kl: f64,
}

impl Default for TrpoPolicyUpdateRule {
    fn default() -> Self {
        Self {
            max_policy_step_kl: 0.01,
        }
    }
}

impl<P, C, O, AS> UpdatePolicyWithOptimizer<P, C, O, AS> for TrpoPolicyUpdateRule
where
    P: SequenceModule + CudnnSupport + ?Sized,
    C: Critic + ?Sized,
    AS: ParameterizedDistributionSpace<Tensor> + ?Sized,
    O: TrustRegionOptimizer + ?Sized,
{
    fn update_policy_with_optimizer(
        &self,
        policy: &P,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        optimizer: &mut O,
        action_space: &AS,
        logger: &mut dyn TimeSeriesLogger,
    ) -> PolicyStats {
        let _cudnn_disable_guard = if policy.has_cudnn_second_derivatives() {
            None
        } else {
            Some(WithCudnnEnabled::new(false))
        };

        let observation_features = features.observation_features();
        let batch_sizes = features.batch_sizes_tensor();
        let actions = features.actions();

        let (step_values, initial_distribution, initial_log_probs, initial_policy_entropy) = {
            let _no_grad = tch::no_grad_guard();

            let step_values = critic.seq_packed(features);
            let policy_output = policy.seq_packed(observation_features, batch_sizes);
            let distribution = action_space.distribution(&policy_output);
            let log_probs = distribution.log_probs(actions);
            let entropy = distribution.entropy().mean(Kind::Float);

            (step_values, distribution, log_probs, f64::from(entropy))
        };

        let policy_loss_distance_fn = || {
            let policy_output = policy.seq_packed(observation_features, batch_sizes);
            let distribution = action_space.distribution(&policy_output);

            let log_probs = distribution.log_probs(actions);
            let likelihood_ratio = (log_probs - &initial_log_probs).exp();
            let loss = -(likelihood_ratio * &step_values).mean(Kind::Float);

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
            &mut logger.event_logger(Event::Epoch),
        );
        if let Err(error) = result {
            match error {
                OptimizerStepError::NaNLoss => panic!("NaN loss in policy optimization"),
                OptimizerStepError::NaNConstraint => {
                    panic!("NaN constraint in policy optimization")
                }
                e => logger
                    .log(Event::Epoch, "no_policy_step", e.to_string().into())
                    .unwrap(),
            }
        }

        PolicyStats {
            entropy: Some(initial_policy_entropy),
        }
    }
}
