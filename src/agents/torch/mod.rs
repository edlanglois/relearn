//! Agents that use torch
mod gae_policy_gradient;
mod history;
mod policy_gradient;

pub use gae_policy_gradient::{GaePolicyGradientAgent, GaePolicyGradientAgentConfig};
pub use history::HistoryBuffer;
pub use policy_gradient::{PolicyGradientAgent, PolicyGradientAgentConfig};
