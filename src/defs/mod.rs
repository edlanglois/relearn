//! Definition structures
//!
//! Each definition is a single type that can specify any of many different types implementing
//! a trait. These types allow the agent and environment to be specified at runtime.
pub mod agent;
mod critic;
pub mod env;
mod hook;
mod optimizer;
mod seq_mod;
mod updater;

pub use agent::{AgentDef, MultiThreadAgentDef};
pub use critic::CriticDef;
pub use env::{BanditMeanRewards, EnvDef};
pub use hook::{HookDef, HooksDef};
pub use optimizer::OptimizerDef;
pub use seq_mod::{PolicyDef, SeqModDef};
pub use updater::{CriticUpdaterDef, PolicyUpdaterDef};
