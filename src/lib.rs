//! A reinforcement learning library.
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::doc_markdown)]
#![warn(clippy::explicit_iter_loop)]
#![warn(clippy::for_kv_map)] // part of warn(clippy::all), specifically style?
#![warn(clippy::missing_const_for_fn)] // has some false positives
#![warn(clippy::needless_borrow)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::redundant_closure_for_method_calls)]
#![warn(clippy::use_self)] // also triggered by macro expansions
pub mod agents;
pub mod cli;
pub mod defs;
pub mod envs;
mod error;
pub mod logging;
pub mod simulation;
pub mod spaces;
pub mod torch;
pub mod utils;

pub use agents::{Actor, Agent, Step};
pub use defs::{AgentDef, EnvDef, OptimizerDef, SeqModDef};
pub use envs::{EnvStructure, Environment, StatefulEnvironment};
pub use error::RLError;
pub use simulation::{run_actor, run_agent, Simulation};
