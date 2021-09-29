//! Reinforcement learning agents using torch
mod actor_critic;
#[cfg(test)]
mod tests;

pub use actor_critic::{ActorCriticAgent, ActorCriticConfig};

use super::{
    critic::Critic,
    policy::Policy,
    updaters::{UpdateCritic, UpdatePolicy},
};

/// Actor critic agent with boxed components.
pub type ActorCriticBoxedAgent<OS, AS> = ActorCriticAgent<
    OS,
    AS,
    Box<dyn Policy>,
    Box<dyn UpdatePolicy<AS>>,
    Box<dyn Critic>,
    Box<dyn UpdateCritic>,
>;
