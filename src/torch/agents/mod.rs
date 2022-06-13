mod actor_critic;
pub mod critics;
pub mod features;
pub mod policies;

pub use actor_critic::{ActorCriticAgent, ActorCriticConfig};

use crate::logging::StatsLogger;
use crate::torch::optimizers::{opt_expect_ok_log, Optimizer};
use std::time::Instant;
use tch::Tensor;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum ToLog {
    /// Don't log the absolute loss value (can log loss changes).
    NoAbsLoss,
    /// Log everything
    All,
}

/// Take n backward steps of a loss function with logging.
fn n_backward_steps<O, F, L>(
    optimizer: &mut O,
    mut loss_fn: F,
    n: u64,
    mut logger: L,
    to_log: ToLog,
    err_msg: &str,
) where
    O: Optimizer + ?Sized,
    F: FnMut() -> Tensor,
    L: StatsLogger,
{
    let mut step_logger = (&mut logger).with_scope("step");
    let mut prev_loss = None;
    let mut prev_start = Instant::now();
    for _ in 0..n {
        let result = optimizer.backward_step(&mut loss_fn, &mut step_logger);
        let loss = opt_expect_ok_log(result, err_msg).map(f64::from);

        if let Some(loss_improvement) = prev_loss.and_then(|p| loss.map(|l| p - l)) {
            step_logger.log_scalar("loss_improvement", loss_improvement);
        }
        prev_loss = loss;
        let end = Instant::now();
        step_logger.log_duration("time", end - prev_start);
        prev_start = end;
    }
    if matches!(to_log, ToLog::All) {
        if let Some(loss) = prev_loss {
            logger.log_scalar("loss", loss);
        }
    }
}
