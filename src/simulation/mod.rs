//! Simulating agent-environment interaction
pub mod hooks;
mod multithread;
mod pair;
mod serial;

pub use hooks::{BuildSimulationHook, GenericSimulationHook, SimulationHook};
pub use multithread::{MultithreadSimulator, MultithreadSimulatorConfig};
pub use pair::PairSimulator;
pub use serial::{run_actor, run_agent, SerialSimulator};

use crate::agents::BuildAgentError;
use crate::envs::{BuildEnvError, PartialSuccessor, Successor};
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

/// Full description of an environment step.
///
/// Includes the successor observation if one exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullStep<O, A> {
    /// The initial observation.
    pub observation: O,
    /// The action taken from the initial state given the initial observation.
    pub action: A,
    /// The resulting reward.
    pub reward: f64,
    /// The next observation or outcome; how the episode progresses.
    pub next: Successor<O>,
}

impl<O, A> FullStep<O, A> {
    pub const fn new(observation: O, action: A, reward: f64, next: Successor<O>) -> Self {
        Self {
            observation,
            action,
            reward,
            next,
        }
    }
}

/// Partial description of an environment step.
///
/// Includes everything except the next observation when the episode continues.
/// Using this instead of [`FullStep`] can avoid having to make copies of the observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfStep<O, A> {
    /// The initial observation.
    pub observation: O,
    /// The action taken from the initial state given the initial observation.
    pub action: A,
    /// The resulting reward.
    pub reward: f64,
    /// The step outcome; how the episode progresses.
    pub next: PartialSuccessor<O>,
}
