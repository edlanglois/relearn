//! Reinforcement learning agents
//!
//! More agents can be found in [crate::torch::agents].
mod agents;
mod bandits;
mod builder;
mod random;
mod tabular;
#[cfg(test)]
pub mod testing;

pub use agents::{Actor, ActorAgent, Agent, Step};
pub use bandits::{
    BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, UCB1Agent, UCB1AgentConfig,
};
pub use builder::{AgentBuilder, BuildAgentError};
pub use random::{RandomAgent, RandomAgentConfig};
pub use tabular::{TabularQLearningAgent, TabularQLearningAgentConfig};
