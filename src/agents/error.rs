//! Agent errors
use tch::TchError;
use thiserror::Error;

/// Error constructing an agent
#[derive(Debug, Error)]
pub enum NewAgentError {
    #[error("space bound(s) are too loose for this agent")]
    InvalidSpaceBounds,
    #[error("reward range must not be unbounded")]
    UnboundedReward,
    #[error(transparent)]
    TorchError(#[from] TchError),
}
