use crate::envs::EnvStructure;
use tch::TchError;
use thiserror::Error;

/// Build an agent instance.
pub trait AgentBuilder<OS, AS> {
    type Agent;

    /// Build an agent for the given environment structure.
    ///
    /// # Args:
    /// `env_structure` - The structure of the environment in which the agent is to operate.
    /// `seed` - A number used to seed the agent's random state,
    ///          for those agents that use deterministic pseudo-random number generation.
    fn build(
        &self,
        env_structure: EnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError>;
}

/// Error building an agent
#[derive(Debug, Error)]
pub enum BuildAgentError {
    #[error("space bound(s) are too loose for this agent")]
    InvalidSpaceBounds,
    #[error("reward range must not be unbounded")]
    UnboundedReward,
    #[error(transparent)]
    TorchError(#[from] TchError),
}
