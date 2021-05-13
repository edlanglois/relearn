//! Error type
use crate::agents::BuildAgentError;
use crate::envs::BuildEnvError;
use thiserror::Error;

/// Error from the Rust RL crate.
#[derive(Error, Debug)]
pub enum RLError {
    #[error("error building agent")]
    BuildAgent(#[from] BuildAgentError),
    #[error("error building environment")]
    BuildEnv(#[from] BuildEnvError),
}
