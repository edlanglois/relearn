//! Reinforcement learning agents using torch
mod policy_gradient;

pub use policy_gradient::{PolicyGradientAgent, PolicyGradientAgentConfig};

use super::seq_modules::StatefulIterSeqModule;
use super::step_value::StepValue;
use tch::COptimizer;

/// Policy gradient with a boxed policy.
pub type PolicyGradientBoxedAgent<OS, AS> = PolicyGradientAgent<
    OS,
    AS,
    Box<dyn StatefulIterSeqModule>,
    COptimizer,
    Box<dyn StepValue>,
    COptimizer,
>;
