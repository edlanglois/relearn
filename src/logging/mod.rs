//! Logging statistics from simulation runs
pub mod cli;

pub use cli::CLILogger;
use std::convert::From;

use enum_map::Enum;
use std::error::Error;
use std::fmt;

/// Simulation run events.
#[derive(Debug, Clone, Copy, PartialEq, Enum)]
pub enum Event {
    Step,
    Episode,
    Epoch,
}

/// A value that can be logged.
#[derive(Debug)]
pub enum Loggable {
    /// Nothing. No data to log.
    /// Logging Nothing data may still produce a placeholder entry for the name.
    Nothing,
    /// A scalar value. Aggregate by taking means.
    Scalar(f64),
    /// A sample from a distrbution over 0 .. (size-1)
    IndexSample { value: usize, size: usize },
}

impl From<f64> for Loggable {
    fn from(value: f64) -> Loggable {
        Loggable::Scalar(value)
    }
}

impl From<f32> for Loggable {
    fn from(value: f32) -> Loggable {
        Loggable::Scalar(value as f64)
    }
}

/// Log statistics from a simulation run.
pub trait Logger {
    /// Log a value.
    ///
    /// # Args
    /// * `event` - The event associated with this value.
    /// * `name` - The name that identifies this value.
    /// * `value` - The value to log.
    ///
    /// # Returns
    /// May return an error if the logged value is structurally incompatible
    /// with previous values logged under the same name.
    fn log<'a>(&mut self, event: Event, name: &'a str, value: Loggable)
        -> Result<(), LogError<'a>>;

    /// Mark the end of an event.
    fn done(&mut self, event: Event);
}

/// Logger that does nothing
impl Logger for () {
    fn log<'a>(&mut self, _: Event, _: &'a str, _: Loggable) -> Result<(), LogError<'a>> {
        Ok(())
    }

    fn done(&mut self, _: Event) {}
}

#[derive(Debug)]
pub struct LogError<'a> {
    name: &'a str,
    value: Loggable,
    expected: String,
}

impl<'a> LogError<'a> {
    pub fn new(name: &'a str, value: Loggable, expected: String) -> Self {
        Self {
            name,
            value,
            expected,
        }
    }
}

impl<'a> fmt::Display for LogError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "\"{}\": incompatible value {:?}, expected {}",
            self.name, self.value, self.expected
        )
    }
}

impl<'a> Error for LogError<'a> {}
