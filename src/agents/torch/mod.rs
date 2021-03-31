//! Agents that use torch
mod policy_gradient;

pub use policy_gradient::{simple_mlp_policy, PolicyGradientAgent};
