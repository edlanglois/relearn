//! Agents that use torch
mod vpg;
mod vpg_gae;

pub use vpg::{PolicyGradientAgent, PolicyGradientAgentConfig};
pub use vpg_gae::{GaePolicyGradientAgent, GaePolicyGradientAgentConfig};
