use super::super::{critic::Critic, history::PackedHistoryFeaturesView, optimizers::Optimizer};
use super::UpdateCriticWithOptimizer;
use crate::logging::{Event, TimeSeriesLogger, TimeSeriesLoggerHelper};

// TODO: Move the MSE training from critic/ to here

/// Rule that updates a critic by minimizing its loss.
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

impl<C, O> UpdateCriticWithOptimizer<C, O> for CriticLossUpdateRule
where
    C: Critic + ?Sized,
    O: Optimizer + ?Sized,
{
    fn update_critic_with_optimizer(
        &self,
        critic: &C,
        features: &dyn PackedHistoryFeaturesView,
        optimizer: &mut O,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        if !critic.trainable() {
            return;
        }
        let loss_fn = || critic.loss(features).unwrap();

        for i in 0..self.optimizer_iters {
            let loss = optimizer
                .backward_step(&loss_fn, &mut logger.event_logger(Event::AgentValueOptStep))
                .unwrap();
            logger.unwrap_log_scalar(Event::AgentValueOptStep, "loss", f64::from(&loss));
            logger.end_event(Event::AgentValueOptStep);

            if i == 0 {
                logger.unwrap_log_scalar(Event::AgentOptPeriod, "initial_loss", loss);
            } else if i == self.optimizer_iters - 1 {
                logger.unwrap_log_scalar(Event::AgentOptPeriod, "final_loss", loss);
            }
        }
    }
}
