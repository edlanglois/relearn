use tch::TchError;
use thiserror::Error;

/// Build an agent instance.
pub trait AgentBuilder<T, E: ?Sized> {
    /// Build an agent for the given environment structure.
    ///
    /// # Args:
    /// `env` - The structure of the environment in which the agent is to operate.
    /// `seed` - A number used to seed the agent's random state,
    ///          for those agents that use deterministic pseudo-random number generation.
    fn build_agent(&self, env: &E, seed: u64) -> Result<T, BuildAgentError>;
}

/// Error building an agent
#[derive(Error, Debug)]
pub enum BuildAgentError {
    #[error("space bound(s) are too loose for this agent")]
    InvalidSpaceBounds,
    #[error("reward range must not be unbounded")]
    UnboundedReward,
    #[error(transparent)]
    TorchError(#[from] TchError),
}
