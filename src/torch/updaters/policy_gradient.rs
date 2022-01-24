//! Policy-gradient policy updater.
use super::super::{
    critic::Critic, features::PackedHistoryFeaturesView, modules::SequenceModule,
    optimizers::Optimizer,
};
use super::{PolicyStats, UpdatePolicyWithOptimizer};
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::ParameterizedDistributionSpace;
use crate::utils::distributions::ArrayDistribution;
use std::cell::Cell;
use tch::{Kind, Tensor};

/// Policy gradient update rule.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PolicyGradientUpdateRule;

impl<O, AS> UpdatePolicyWithOptimizer<O, AS> for PolicyGradientUpdateRule
where
    O: Optimizer + ?Sized,
    AS: ParameterizedDistributionSpace<Tensor> + ?Sized,
{
    fn update_policy_with_optimizer(
        &self,
        policy: &dyn SequenceModule,
        critic: &dyn Critic,
        features: &dyn PackedHistoryFeaturesView,
        optimizer: &mut O,
        action_space: &AS,
        logger: &mut dyn TimeSeriesLogger,
    ) -> PolicyStats {
        logger.start_event(Event::AgentPolicyOptStep).unwrap();
        let step_values = tch::no_grad(|| critic.seq_packed(features));

        let action_dist_params = policy.seq_packed(
            features.observation_features(),
            features.batch_sizes_tensor(),
        );

        let entropies = Cell::new(None);
        let policy_loss_fn = || {
            let action_distributions = action_space.distribution(&action_dist_params);
            let log_probs = action_distributions.log_probs(features.actions());
            entropies.set(Some(action_distributions.entropy()));
            -(log_probs * &step_values).mean(Kind::Float)
        };

        let _ = optimizer
            .backward_step(
                &policy_loss_fn,
                &mut logger.event_logger(Event::AgentPolicyOptStep),
            )
            .unwrap();
        logger.end_event(Event::AgentPolicyOptStep).unwrap();

        let entropy = entropies.into_inner().unwrap().mean(Kind::Float).into();
        PolicyStats {
            entropy: Some(entropy),
        }
    }
}
