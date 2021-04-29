//! Reinforcement learning agents
mod agents;
mod bandits;
mod builder;
mod random;
mod tabular;
#[cfg(test)]
mod testing;
mod torch;

pub use agents::{Actor, ActorAgent, Agent, Step};
pub use bandits::{
    BetaThompsonSamplingAgent, BetaThompsonSamplingAgentConfig, UCB1Agent, UCB1AgentConfig,
};
pub use builder::{AgentBuilder, BuildAgentError};
pub use random::{RandomAgent, RandomAgentConfig};
pub use tabular::{TabularQLearningAgent, TabularQLearningAgentConfig};
pub use torch::step_value;
pub use torch::{PolicyGradientAgent, PolicyGradientAgentConfig};

use crate::torch::seq_modules::StatefulIterSeqModule;
use tch::COptimizer;
use torch::step_value::StepValue;

/// Policy gradient with a boxed policy.
pub type PolicyGradientBoxedAgent<OS, AS> = PolicyGradientAgent<
    OS,
    AS,
    Box<dyn StatefulIterSeqModule>,
    COptimizer,
    Box<dyn StepValue>,
    COptimizer,
>;
