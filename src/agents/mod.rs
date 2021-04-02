//! Reinforcement learning agents
mod agents;
mod bandits;
pub mod error;
mod random;
mod tabular;
#[cfg(test)]
mod testing;
mod torch;

pub use agents::{Actor, Agent, Step};
pub use bandits::{BetaThompsonSamplingAgent, UCB1Agent};
pub use random::RandomAgent;
pub use tabular::TabularQLearningAgent;
pub use torch::{MLPConfig, PolicyGradientAgent};
