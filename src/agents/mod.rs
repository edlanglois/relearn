//! Reinforcement learning agents
mod agents;
mod bandits;
mod builder;
mod error;
mod random;
mod tabular;
#[cfg(test)]
mod testing;
mod torch;

pub use agents::{Actor, Agent, Step};
pub use bandits::{
    BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, UCB1Agent, UCB1AgentConfig,
};
pub use builder::AgentBuilder;
pub use error::NewAgentError;
pub use random::{RandomAgent, RandomAgentConfig};
pub use tabular::{TabularQLearningAgent, TabularQLearningAgentConfig};
pub use torch::{PolicyGradientAgent, PolicyGradientAgentConfig};
