//! Simulating agent-environment interaction
pub mod hooks;
mod parallel;
mod serial;

pub use hooks::{BuildStructuredHook, GenericSimulationHook, SimulationHook};
pub use parallel::{run_agent_multithread, ParallelSimulator, ParallelSimulatorConfig};
pub use serial::{run_actor, run_agent, SerialSimulator};

use crate::agents::BuildAgentError;
use crate::envs::BuildEnvError;
use crate::logging::TimeSeriesLogger;
use thiserror::Error;

/// Runs agent-environment simulations.
pub trait Simulator {
    /// Run a simulation
    ///
    /// # Args
    /// * `env_seed` - Random seed for generating the environment instance or instances.
    ///                Environment instances use the seeds `env_seed`, `env_seed + 1`, etc.
    /// * `agent_seed` - Random seed for initializing the agent or agent workers.
    ///                Agnet workers use the seeds `agent_seed`, `agent_seed + 1`, etc.
    /// * `logger` - The logger for the main thread.
    fn run_simulation(
        &mut self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), SimulatorError>;
}

/// Error initializing or running a simulation.
#[derive(Error, Debug)]
pub enum SimulatorError {
    #[error("error building agent")]
    BuildAgent(#[from] BuildAgentError),
    #[error("error building environment")]
    BuildEnv(#[from] BuildEnvError),
}
