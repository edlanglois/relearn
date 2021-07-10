//! Proximal policy optimization policy updater.
use super::super::{
    critic::Critic, history::PackedHistoryFeaturesView, optimizers::Optimizer,
    seq_modules::SequenceModule,
};
use super::{PolicyStats, UpdatePolicyWithOptimizer};
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::ParameterizedDistributionSpace;
use crate::utils::distributions::ArrayDistribution;
use tch::{Kind, Tensor};

/// Proximal policy optimization update rule with a clipped objective.
///
/// # Reference
/// [Proximal Policy Optimization Algorithms][ppo] by Schulman et al.
///
/// [ppo]: https://arxiv.org/abs/1707.06347
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PpoPolicyUpdateRule {
    // pub minibatch_size: usize, // TODO: Support minibatches
    pub num_epochs: u64,
    pub clip_distance: f64,
}

impl Default for PpoPolicyUpdateRule {
    fn default() -> Self {
        Self {
            num_epochs: 10,
            clip_distance: 0.2,
        }
    }
}

impl<P, C, O, AS> UpdatePolicyWithOptimizer<P, C, O, AS> for PpoPolicyUpdateRule
where
    P: SequenceModule + ?Sized,
    C: Critic + ?Sized,
    O: Optimizer + ?Sized,
    AS: ParameterizedDistributionSpace<Tensor> + ?Sized,
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
        let observation_features = features.observation_features();
        let batch_sizes = features.batch_sizes_tensor();
        let actions = features.actions();

        let (step_values, initial_log_probs, initial_policy_entropy) = {
            let _no_grad = tch::no_grad_guard();

            let step_values = critic.seq_packed(features);
            let policy_output = policy.seq_packed(observation_features, batch_sizes);
            let distribution = action_space.distribution(&policy_output);
            let log_probs = distribution.log_probs(actions);
            let entropy = distribution.entropy().mean(Kind::Float);

            (step_values, log_probs, f64::from(entropy))
        };

        let policy_surrogate_loss_fn = || {
            let policy_output = policy.seq_packed(observation_features, batch_sizes);
            let distribution = action_space.distribution(&policy_output);
            let log_probs = distribution.log_probs(actions);

            let likelihood_ratio = (log_probs - &initial_log_probs).exp();
            let clipped_likelihood_ratio =
                likelihood_ratio.clip(1.0 - self.clip_distance, 1.0 + self.clip_distance);

            (likelihood_ratio * &step_values)
                .min_other(&(clipped_likelihood_ratio * &step_values))
                .mean(Kind::Float)
                .neg()
        };

        let _ = optimizer
            .backward_step(
                &policy_surrogate_loss_fn,
                &mut logger.event_logger(Event::AgentPolicyOptStep),
            )
            .unwrap();
        logger.end_event(Event::AgentPolicyOptStep);

        PolicyStats {
            entropy: Some(initial_policy_entropy),
        }
    }
}
