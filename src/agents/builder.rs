use super::{Actor, Agent, NewAgentError};
use crate::envs::EnvStructure;
use crate::spaces::Space;

/// Build an agent instance.
pub trait AgentBuilder<OS: Space, AS: Space> {
    type Agent: Actor<OS::Element, AS::Element> + Agent<OS::Element, AS::Element>;

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
    ) -> Result<Self::Agent, NewAgentError>;
}
