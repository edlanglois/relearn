//! Agents that use torch
mod policy_gradient;

pub use policy_gradient::{PolicyGradientAgent, PolicyGradientAgentConfig};

use crate::torch::seq_modules::{IterativeModule, SequenceModule};

/// A Torch policy
pub trait Policy: SequenceModule + IterativeModule {}

impl<T: SequenceModule + IterativeModule> Policy for T {}
