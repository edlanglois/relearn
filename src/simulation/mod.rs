//! Simulating agent-environment interaction
pub mod hooks;
mod iter;
mod multithread;
mod serial;

pub use hooks::{BuildSimulationHook, SimulationHook};
pub use iter::SimSteps;
pub use multithread::{MultithreadSimulator, MultithreadSimulatorConfig};
pub use serial::SerialSimulator;

use crate::agents::{BuildAgentError, SynchronousAgent};
use crate::envs::{BuildEnvError, Environment, Successor};
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

/// Description of an environment step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step<O, A, U = O> {
    /// The initial observation.
    pub observation: O,
    /// The action taken from the initial state given the initial observation.
    pub action: A,
    /// The resulting reward.
    pub reward: f64,
    /// The next observation or outcome; how the episode progresses.
    pub next: Successor<O, U>,
}

impl<O, A, U> Step<O, A, U> {
    pub const fn new(observation: O, action: A, reward: f64, next: Successor<O, U>) -> Self {
        Self {
            observation,
            action,
            reward,
            next,
        }
    }

    pub fn into_partial(self) -> PartialStep<O, A> {
        Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: self.next.into_partial(),
        }
    }
}

/// Description of an environment step where the successor observation is borrowed.
pub type TransientStep<'a, O, A> = Step<O, A, &'a O>;

impl<O: Clone, A> TransientStep<'_, O, A> {
    /// Convert a transient step into an owned step by cloning any borrowed successor observation.
    #[inline]
    pub fn into_owned(self) -> Step<O, A> {
        Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: self.next.into_owned(),
        }
    }
}

/// Partial description of an environment step.
///
/// The successor state is omitted when the episode continues.
/// Using this can help avoid copying the observation.
pub type PartialStep<O, A> = Step<O, A, ()>;

/// Run an agent-environment simulation.
///
/// Note that `Environment`, `SynchronousAgent`, etc. are also implemented for mutable references
/// so this function can be called either with owned objects or with references.
///
/// # Args
/// * `environment` - The environment to simulate.
/// * `agent` - The agent to simulate.
/// * `hook` - A simulation hook run on each step. Controls when the simulation stops.
/// * `logger` - The logger to use.
pub fn run_agent<E, A, H, L>(environment: E, agent: A, mut hook: H, mut logger: L) -> A
where
    E: Environment,
    A: SynchronousAgent<E::Observation, E::Action>,
    H: SimulationHook<E::Observation, E::Action>,
    L: TimeSeriesLogger,
{
    if !hook.start(&mut logger) {
        return agent;
    }
    let mut sim = environment.run(agent, logger);
    while sim.step_with(|_, agent, step, logger| {
        let continue_ = hook.call(&step, logger);
        agent.update(step, logger);
        continue_
    }) {}
    sim.actor
}
