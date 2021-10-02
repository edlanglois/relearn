//! Simulating agent-environment interaction
pub mod hooks;
mod parallel;
mod serial;

pub use hooks::{BuildStructuredHook, GenericSimulationHook, SimulationHook};
pub use parallel::{run_agent_multithread, MultiThreadSimulator, MultiThreadSimulatorConfig};
pub use serial::{run_actor, run_agent, Simulator, SimulatorConfig};

use crate::envs::BuildEnvError;
use crate::logging::TimeSeriesLogger;
use thiserror::Error;

/// Runs a simulation.
pub trait RunSimulation {
    /// Run a simulation
    fn run_simulation(&mut self, logger: &mut dyn TimeSeriesLogger);
}

/// Error building a simulator
#[derive(Debug, Error)]
pub enum BuildSimError {
    #[error(transparent)]
    EnvError(#[from] BuildEnvError),
}
