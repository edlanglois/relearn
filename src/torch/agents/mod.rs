//! Reinforcement learning agents using torch
mod actor_critic;
// XXX Uncomment once batch functionality has been restored
// #[cfg(test)]
// mod tests;

pub use actor_critic::{ActorCriticAgent, ActorCriticConfig};
