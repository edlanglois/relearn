pub mod cli;

pub use cli::CLILogger;

use enum_map::Enum;

/// Simulation run events.
#[derive(Debug, Clone, Copy, PartialEq, Enum)]
pub enum Event {
    Step,
    Episode,
    Epoch,
}

/// Log statistics from a simulation run.
pub trait Logger {
    /// Log a floating-point scalar value.
    fn log_scalar(&mut self, event: Event, name: &str, value: f64);

    /// Mark the end of the given event.
    fn done(&mut self, event: Event);

    /// Terminate the logger, ensuring all logs are flushed
    fn close(&mut self);
}
