use super::super::{critic::Critic, features::PackedHistoryFeaturesView, optimizers::Optimizer};
use super::UpdateCriticWithOptimizer;
use crate::logging::StatsLogger;
use std::time::Instant;

// TODO: Move the MSE training from critic/ to here

/// Rule that updates a critic by minimizing its loss for several optimizer iterations.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CriticLossUpdateRule {
    /// Number of optimizer iterations per update
    pub optimizer_iters: u64,
}

impl Default for CriticLossUpdateRule {
    fn default() -> Self {
        Self {
            optimizer_iters: 80,
        }
    }
}

impl<O> UpdateCriticWithOptimizer<O> for CriticLossUpdateRule
where
    O: Optimizer + ?Sized,
{
    fn update_critic_with_optimizer(
        &self,
        critic: &dyn Critic,
        features: &dyn PackedHistoryFeaturesView,
        optimizer: &mut O,
        logger: &mut dyn StatsLogger,
    ) {
        if !critic.trainable() {
            return;
        }
        let loss_fn = || critic.loss(features).unwrap();

        let mut critic_opt_start = Instant::now();
        let mut initial_loss = None;
        let mut final_loss = None;
        for i in 0..self.optimizer_iters {
            let loss = f64::from(optimizer.backward_step(&loss_fn, logger).unwrap());
            let critic_opt_end = Instant::now();
            logger.log_scalar("step/loss", loss);
            logger.log_counter_increment("step/count", 1);
            logger.log_duration("step/time", critic_opt_end - critic_opt_start);
            critic_opt_start = critic_opt_end;

            if i == 0 {
                initial_loss = Some(loss);
            } else if i == self.optimizer_iters - 1 {
                final_loss = Some(loss);
            }
        }
        logger.log_scalar(
            "loss_improvement",
            initial_loss.unwrap() - final_loss.unwrap(),
        )
    }
}
