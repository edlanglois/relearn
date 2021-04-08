//! Agents that use torch
mod vpg;
mod vpg_gae;

pub use vpg::{PolicyGradientAgent, PolicyGradientAgentConfig};
pub use vpg_gae::{GaePolicyGradientAgent, GaePolicyGradientAgentConfig};

use crate::torch::seq_modules::{IterativeModule, SequenceModule};

/// A Torch policy
pub trait Policy: SequenceModule + IterativeModule {}

impl<T: SequenceModule + IterativeModule> Policy for T {}
