//! Agent errors
use std::fmt;
use thiserror::Error;

/// Error constructing an agent
#[derive(Debug, Error)]
pub enum NewAgentError {
    #[error(
        "invalid observation space {observation_space:?} and/or action space {action_space:?}"
    )]
    InvalidSpace {
        observation_space: Box<dyn fmt::Debug>,
        action_space: Box<dyn fmt::Debug>,
    },
    #[error("reward range must not be unbounded")]
    UnboundedReward,
}
