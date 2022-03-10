//! Reinforcement learning agents using torch
mod actor_critic;
#[cfg(test)]
mod tests;

pub use actor_critic::{ActorCriticActor, ActorCriticAgent, ActorCriticConfig};
