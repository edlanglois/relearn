//! Simulating agent-environment interaction
pub mod hooks;
mod multithread;
mod serial;

pub use hooks::{BuildSimulationHook, GenericSimulationHook, SimulationHook};
pub use multithread::{MultithreadSimulator, MultithreadSimulatorConfig};
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
    /// * `env_seed` - Random seed used to derive the environment initialization seed(s).
    /// * `agent_seed` - Random seed used to derive the agent initialization seed(s).
    /// * `logger` - The logger for the main thread.
    fn run_simulation(
        &self,
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
