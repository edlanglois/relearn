//! Simulating agent-environment interaction
pub mod hooks;
mod serial;

pub use hooks::{GenericSimulationHook, SimulationHook};
pub use serial::{run_actor, run_agent, Simulator};

/// Runs a simulation.
pub trait RunSimulation {
    /// Run a simulation
    fn run_simulation(&mut self);
}
