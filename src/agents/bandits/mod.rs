//! Multi-armed bandit agents.
//!
//! These agents do not model any relationship between states.
mod thompson_sampling;

pub use thompson_sampling::BetaThompsonSamplingAgent;
