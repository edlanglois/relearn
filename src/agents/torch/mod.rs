//! Agents that use torch
mod history;
mod vpg;
mod vpg_gae;

pub use history::HistoryBuffer;
pub use vpg::{PolicyGradientAgent, PolicyGradientAgentConfig};
pub use vpg_gae::{GaePolicyGradientAgent, GaePolicyGradientAgentConfig};
