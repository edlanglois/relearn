//! Reinforcement learning agents using torch
mod actor_critic;
#[cfg(test)]
mod tests;

pub use actor_critic::{ActorCriticAgent, ActorCriticConfig};

use super::{
    backends::CudnnSupport,
    critic::Critic,
    seq_modules::{SequenceModule, StatefulIterativeModule},
    updaters::{UpdateCritic, UpdatePolicy},
};
use tch::Tensor;

// TODO: Remove Box<> inside UpdatePolicy & UpdateCritic

/// Actor critic agent with boxed components.
pub type ActorCriticBoxedAgent<OS, AS> = ActorCriticAgent<
    OS,
    AS,
    Box<dyn ACPolicyModule>,
    Box<dyn UpdatePolicy<Box<dyn ACPolicyModule>, Box<dyn Critic>, AS>>,
    Box<dyn Critic>,
    Box<dyn UpdateCritic<Box<dyn Critic>>>,
>;

/// Unified policy module trait required by [`ActorCriticBoxedAgent`].
pub trait ACPolicyModule: SequenceModule + StatefulIterativeModule + CudnnSupport {}
impl<T: SequenceModule + StatefulIterativeModule + CudnnSupport + ?Sized> ACPolicyModule for T {}

box_impl_sequence_module!(dyn ACPolicyModule);
box_impl_stateful_iterative_module!(dyn ACPolicyModule);
box_impl_cudnn_support!(dyn ACPolicyModule);
