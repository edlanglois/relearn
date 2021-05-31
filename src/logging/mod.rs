//! Logging statistics from simulation runs
#![allow(clippy::use_self)] // false positive in Enum derive for Event
pub mod cli;

pub use cli::CLILogger;
use enum_map::Enum;
use std::borrow::Cow;
use std::convert::From;
use std::error::Error;
use std::fmt;

/// Simulation run events types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum Event {
    Step,
    Episode,
    Epoch,
}

/// A value that can be logged.
#[derive(Debug, Clone)]
pub enum Loggable {
    /// Nothing. No data to log.
    /// Logging Nothing data may still produce a placeholder entry for the name.
    Nothing,
    /// A scalar value. Aggregate by taking means.
    Scalar(f64),
    /// A sample from a distrbution over 0 .. (size-1)
    IndexSample { value: usize, size: usize },
    /// A message string.
    Message(Cow<'static, str>),
}

impl From<f64> for Loggable {
    fn from(value: f64) -> Self {
        Self::Scalar(value)
    }
}

impl From<f32> for Loggable {
    fn from(value: f32) -> Self {
        Self::Scalar(value.into())
    }
}

impl From<&'static str> for Loggable {
    fn from(value: &'static str) -> Self {
        Self::Message(value.into())
    }
}

impl From<String> for Loggable {
    fn from(value: String) -> Self {
        Self::Message(value.into())
    }
}

/// Log statistics from a simulation run.
pub trait Logger {
    /// Log a value.
    ///
    /// # Args
    /// * `event` - The event type associated with this value.
    /// * `name` - The name that identifies this value.
    /// * `value` - The value to log.
    ///
    /// Each logged value is associated with an event type and a name.
    /// When multiple values are logged for the same event type and name during an event,
    /// only the last is used.
    /// Values may be aggregated across events in a logger-dependent fashion.
    ///
    /// An "event" refers to the period between calls to `Logger::done` for that event type.
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

#[derive(Debug, Clone)]
pub struct LogError<'a> {
    name: &'a str,
    value: Loggable,
    expected: String,
}

impl<'a> LogError<'a> {
    pub const fn new(name: &'a str, value: Loggable, expected: String) -> Self {
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
