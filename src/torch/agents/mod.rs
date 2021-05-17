//! Reinforcement learning agents using torch
mod actor;
mod policy_gradient;
mod trpo;

pub use actor::{PolicyValueNetActor, PolicyValueNetActorConfig};
pub use policy_gradient::{PolicyGradientAgent, PolicyGradientAgentConfig};
pub use trpo::{TrpoAgent, TrpoAgentConfig};

use super::seq_modules::StatefulIterSeqModule;
use super::step_value::StepValue;
use crate::torch::optimizers::ConjugateGradientOptimizer;
use tch::COptimizer;

/// Policy gradient agent with a boxed policy and value function.
pub type PolicyGradientBoxedAgent<OS, AS> = PolicyGradientAgent<
    OS,
    AS,
    Box<dyn StatefulIterSeqModule>,
    COptimizer,
    Box<dyn StepValue>,
    COptimizer,
>;

/// TRPO agent with a boxed policy and value function.
pub type TrpoBoxedAgent<OS, AS> = TrpoAgent<
    OS,
    AS,
    Box<dyn StatefulIterSeqModule>,
    ConjugateGradientOptimizer,
    Box<dyn StepValue>,
    COptimizer,
>;
