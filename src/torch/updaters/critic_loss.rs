use super::super::{critic::Critic, history::PackedHistoryFeaturesView, optimizers::Optimizer};
use super::UpdateCriticWithOptimizer;
use crate::logging::{Event, Logger, TimeSeriesLogger};

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
        let mut logger = logger.event_logger(Event::AgentOptPeriod);

        let loss_fn = || critic.loss(features).unwrap();

        for i in 0..self.optimizer_iters {
            let loss = optimizer.backward_step(&loss_fn, &mut logger).unwrap();
            if i == 0 {
                logger.log("initial_loss", f64::from(loss).into()).unwrap();
            } else if i == self.optimizer_iters - 1 {
                logger.log("final_loss", f64::from(loss).into()).unwrap();
            }
        }
    }
}
