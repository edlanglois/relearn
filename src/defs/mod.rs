//! Definition structures
//!
//! Each definition is a single type that can specify any of many different types implementing
//! a trait. These types allow the agent and environment to be specified at runtime.
mod agent;
pub mod env;
mod optimizer;
mod seq_mod;
mod step_value;

pub use agent::AgentDef;
pub use env::EnvDef;
pub use optimizer::OptimizerDef;
pub use seq_mod::SeqModDef;
pub use step_value::StepValueDef;
