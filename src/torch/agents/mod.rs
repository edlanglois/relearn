//! Reinforcement learning agents using torch
mod actor;
mod policy_gradient;
mod trpo;

pub use actor::{PolicyValueNetActor, PolicyValueNetActorConfig};
pub use policy_gradient::{PolicyGradientAgent, PolicyGradientAgentConfig};
pub use trpo::{TrpoAgent, TrpoAgentConfig};

use super::critic::Critic;
use crate::torch::{
    backends::CudnnSupport,
    optimizers::ConjugateGradientOptimizer,
    seq_modules::{SequenceModule, StatefulIterSeqModule, StatefulIterativeModule},
};
use tch::{COptimizer, Tensor};

/// Policy gradient agent with a boxed policy and value function.
pub type PolicyGradientBoxedAgent<OS, AS> = PolicyGradientAgent<
    OS,
    AS,
    Box<dyn StatefulIterSeqModule>,
    COptimizer,
    Box<dyn Critic>,
    COptimizer,
>;

/// TRPO agent with a boxed policy and value function.
pub type TrpoBoxedAgent<OS, AS> = TrpoAgent<
    OS,
    AS,
    Box<dyn TrpoPolicyModule>,
    ConjugateGradientOptimizer,
    Box<dyn Critic>,
    COptimizer,
>;

/// Unified policy module trait required by [`TrpoBoxedAgent`].
pub trait TrpoPolicyModule: SequenceModule + StatefulIterativeModule + CudnnSupport {}
impl<T: SequenceModule + StatefulIterativeModule + CudnnSupport + ?Sized> TrpoPolicyModule for T {}

box_impl_sequence_module!(dyn TrpoPolicyModule);
box_impl_stateful_iterative_module!(dyn TrpoPolicyModule);
box_impl_cudnn_support!(dyn TrpoPolicyModule);
