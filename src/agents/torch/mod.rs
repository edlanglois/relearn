//! Agents that use torch
mod history;
mod policy_gradient;
pub mod step_value;

pub use history::HistoryBuffer;
pub use policy_gradient::{PolicyGradientAgent, PolicyGradientAgentConfig};
