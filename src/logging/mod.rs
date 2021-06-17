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

// TODO: Make generic over Event? E: Enum

/// Logs named values associated with a time series of recurring events.
///
/// # Events
/// An event represents the duration of time that starts when the last instance of that event
/// ended (or when the logger was created) and ends `TimeSeriesLogger::end_event` is called.
/// In this way, each event type creates a partitining of time into instances of that event.
///
/// Examples of events include an environment step, an environment episode, or a training epoch.
///
/// Values are logged to a particular event type and become associated with the currently active
/// instance of that event. Logging the same name multiple times during an event is allowed but
/// discouraged; prefer using a finer grained event instead. If the same value _is_ logged
/// multiple times in an event then the logger may behave in an implementation-dependent way
/// like logging both values or using only the last.
///
/// # Values
/// Logged values must be of type [`Loggable`].
/// The same variant must be used every time a value is logged for the same name.
pub trait TimeSeriesLogger {
    /// Log a value associated with the active instance of an event.
    ///
    /// # Args
    /// * `event` - The event type associated with this value.
    /// * `name` - The name that identifies this value.
    /// * `value` - The value to log.
    ///
    /// An "event" refers to the period between calls to `TimeSeriesLogger::end_event` for that event type.
    ///
    /// # Returns
    /// May return an error if the logged value is structurally incompatible
    /// with previous values logged under the same name.
    fn log<'a>(&mut self, event: Event, name: &'a str, value: Loggable)
        -> Result<(), LogError<'a>>;

    /// End an event instance.
    fn end_event(&mut self, event: Event);
}

/// Time series logger that does nothing
impl TimeSeriesLogger for () {
    fn log<'a>(&mut self, _: Event, _: &'a str, _: Loggable) -> Result<(), LogError<'a>> {
        Ok(())
    }

    fn end_event(&mut self, _: Event) {}
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
