//! Multi-armed bandit agents.
//!
//! These agents do not model any relationship between states.
mod thompson_sampling;
mod ucb;

pub use thompson_sampling::BetaThompsonSamplingAgent;
pub use ucb::UCB1Agent;
