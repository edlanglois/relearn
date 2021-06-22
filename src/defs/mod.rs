//! Definition structures
//!
//! Each definition is a single type that can specify any of many different types implementing
//! a trait. These types allow the agent and environment to be specified at runtime.
pub mod agent;
mod critic;
pub mod env;
mod optimizer;
mod seq_mod;

pub use agent::AgentDef;
pub use critic::CriticDef;
pub use env::EnvDef;
pub use optimizer::OptimizerDef;
pub use seq_mod::SeqModDef;
