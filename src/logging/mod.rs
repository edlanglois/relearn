//! Logging statistics from simulation runs
#![allow(clippy::use_self)] // false positive in Enum derive for Event
mod cli;

pub use cli::CLILogger;
use enum_map::Enum;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::convert::{AsRef, From, Into};
use std::fmt;
use thiserror::Error;

/// Simulation run events types.
///
/// Each describes a recurring event that partitions time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum Event {
    /// Environment step
    EnvStep,
    /// Environment episode
    EnvEpisode,
    /// Agent policy optimization step
    ///
    /// One parameter update step using a (mini)-batch of data.
    AgentPolicyOptStep,
    /// Agent value optimization step
    ///
    /// One parameter update step using a (mini)-batch of data.
    AgentValueOptStep,
    /*
    // TODO: Use or remove
    /// Agent optimization epoch
    ///
    /// A set of optimization steps that collectively perform a single pass
    /// through all training data for the period.
    AgentOptEpoch,
    */
    /// Agent optimization period
    ///
    /// For synchronous agents with a collect-data-then-optimize-parameters loop,
    /// this is one iteration of that loop.
    ///
    /// The name "period" is, to my knowledge, non-standard but extends the geological time scale
    /// analogy of "epoch" and seems appropriate for the periodic nature of synchronous RL agents.
    AgentOptPeriod,
}

/// A value that can be logged.
#[derive(Debug, Clone, PartialEq)]
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

/// Log named values
pub trait Logger {
    /// Log a value
    ///
    /// # Args
    /// * `name` - A name identifying the value.
    /// * `value` - The value to log.
    fn log(&mut self, name: &'static str, value: Loggable) -> Result<(), LogError> {
        self.id_log(name.into(), value)
    }

    /// Log a value for a given hierarchical ID.
    ///
    /// # Args
    /// * `id` - The identifier for the value.
    /// * `value` - The value to log.
    fn id_log(&mut self, id: Id, value: Loggable) -> Result<(), LogError>;

    /// Create a view on the logger that adds a scope prefix to all logged ids.
    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn Logger>;
}

/// Generic helper methods for logging
pub trait LoggerHelper: Logger {
    #[inline]
    fn unwrap_log_scalar<T: Into<f64>>(&mut self, name: &'static str, value: T) {
        self.unwrap_log(name, value.into());
    }
    #[inline]
    fn unwrap_log<T: Into<Loggable>>(&mut self, name: &'static str, value: T) {
        self.log(name, value.into()).unwrap();
    }
}

impl<T: Logger + ?Sized> LoggerHelper for T {}

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
    /// An "event" refers to the period between calls to [`TimeSeriesLogger::end_event`]
    /// for that event type.
    ///
    /// # Returns
    /// May return an error if the logged value is structurally incompatible
    /// with previous values logged under the same name.
    fn log(&mut self, event: Event, name: &'static str, value: Loggable) -> Result<(), LogError> {
        self.id_log(event, name.into(), value)
    }

    /// Log a value for a given hierarchical id.
    ///
    /// See [`TimeSeriesLogger::log`] for details.
    fn id_log(&mut self, event: Event, id: Id, value: Loggable) -> Result<(), LogError>;

    /// End an event instance.
    fn end_event(&mut self, event: Event);

    /// Creates a wrapper [`Logger`] that logs all values to a specific event type.
    ///
    /// This does not create or end event instances,
    /// it just creates a wrapper around [`TimeSeriesLogger::log`] with a fixed event.
    fn event_logger(&mut self, event: Event) -> TimeSeriesEventLogger;

    /// Create a view on the logger that adds a scope prefix to all logged ids.
    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn TimeSeriesLogger>;
}

/// Generic helper methods for logging
pub trait TimeSeriesLoggerHelper: TimeSeriesLogger {
    #[inline]
    fn unwrap_log_scalar<T: Into<f64>>(&mut self, event: Event, name: &'static str, value: T) {
        self.unwrap_log(event, name, value.into());
    }
    #[inline]
    fn unwrap_log<T: Into<Loggable>>(&mut self, event: Event, name: &'static str, value: T) {
        self.log(event, name, value.into()).unwrap()
    }
}
impl<T: TimeSeriesLogger + ?Sized> TimeSeriesLoggerHelper for T {}

pub struct TimeSeriesEventLogger<'a> {
    logger: &'a mut (dyn TimeSeriesLogger + 'a),
    event: Event,
}

impl<'a> Logger for TimeSeriesEventLogger<'a> {
    fn id_log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        self.logger.id_log(self.event, id, value)
    }

    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn Logger> {
        ScopedLogger::new(self, scope)
    }
}

/// Wraps a logger by adding a scope (prefix) to all ids.
pub struct ScopedLogger<'a, L: 'a + ?Sized> {
    logger: &'a mut L,
    prefix: Id,
}

impl<'a, L: ?Sized> ScopedLogger<'a, L> {
    pub fn new<S: Into<Cow<'static, str>>>(logger: &'a mut L, scope: S) -> Self {
        Self {
            logger,
            prefix: Id::from(scope),
        }
    }
}

impl<'a> Logger for ScopedLogger<'a, dyn Logger> {
    fn log(&mut self, name: &'static str, value: Loggable) -> Result<(), LogError> {
        let mut full_id = self.prefix.clone();
        full_id.push(name.into());
        self.logger.id_log(full_id, value)
    }

    fn id_log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        let mut full_id = self.prefix.clone();
        full_id.append(id);
        self.logger.id_log(full_id, value)
    }

    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn Logger> {
        // Copy and extend the prefix rather than wrapper the existing ScopedLogger
        // This way, a nested scope is represented by a single ScopedLogger not a tower of
        // ScopedLoggers that would each create a copy of the partial scope when invoked.
        let mut prefix = self.prefix.clone();
        prefix.push(scope.into());
        ScopedLogger {
            logger: self.logger,
            prefix,
        }
    }
}

impl<'a> TimeSeriesLogger for ScopedLogger<'a, dyn TimeSeriesLogger> {
    fn log(&mut self, event: Event, name: &'static str, value: Loggable) -> Result<(), LogError> {
        let mut full_id = self.prefix.clone();
        full_id.push(name.into());
        self.id_log(event, full_id, value)
    }

    fn id_log(&mut self, event: Event, id: Id, value: Loggable) -> Result<(), LogError> {
        let mut full_id = self.prefix.clone();
        full_id.append(id);
        self.logger.id_log(event, full_id, value)
    }

    fn end_event(&mut self, event: Event) {
        self.logger.end_event(event);
    }

    fn event_logger(&mut self, event: Event) -> TimeSeriesEventLogger {
        TimeSeriesEventLogger {
            logger: self,
            event,
        }
    }

    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn TimeSeriesLogger> {
        // Copy and extend the prefix rather than wrapper the existing ScopedLogger
        // This way, a nested scope is represented by a single ScopedLogger not a tower of
        // ScopedLoggers that would each create a copy of the partial scope when invoked.
        let mut prefix = self.prefix.clone();
        prefix.push(scope.into());
        ScopedLogger {
            logger: self.logger,
            prefix,
        }
    }
}

/// Logger that does nothing
impl Logger for () {
    fn id_log(&mut self, _id: Id, _value: Loggable) -> Result<(), LogError> {
        Ok(())
    }
    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn Logger> {
        ScopedLogger::new(self, scope)
    }
}

/// Time series logger that does nothing
impl TimeSeriesLogger for () {
    fn id_log(&mut self, _event: Event, _id: Id, _value: Loggable) -> Result<(), LogError> {
        Ok(())
    }

    fn end_event(&mut self, _: Event) {}

    fn event_logger(&mut self, event: Event) -> TimeSeriesEventLogger {
        TimeSeriesEventLogger {
            logger: self,
            event,
        }
    }

    fn scope(&mut self, scope: &'static str) -> ScopedLogger<dyn TimeSeriesLogger> {
        ScopedLogger::new(self, scope)
    }
}

/// A hierarchical identifier.
///
/// The identifier consists of the top-level namespace first,
/// followed by inner namespaces, then the identifier name.
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Id(SmallVec<[Cow<'static, str>; 6]>);

impl Id {
    /// Create a new `Id` instance.
    ///
    /// # Errors
    /// If the given name hierarchy does not have at least one element (the name).
    pub fn new<T: Into<SmallVec<[Cow<'static, str>; 6]>>>(names: T) -> Result<Self, LogError> {
        let names = names.into();
        if names.is_empty() {
            return Err(LogError::InvalidEmptyId);
        }
        Ok(Self(names))
    }

    pub fn from_name<T: Into<Cow<'static, str>>>(name: T) -> Self {
        Self((&[name.into()] as &[_]).into())
    }

    /// Append another ID onto this one creating a more deeply nested namespace.
    ///
    /// The new ID is placed inside the namespace created by self.
    pub fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0)
    }

    /// Push a name into the namespace. The new name is on the inside.
    pub fn push(&mut self, name: Cow<'static, str>) {
        self.0.push(name)
    }
}

impl<T: Into<Cow<'static, str>>> From<T> for Id {
    fn from(name: T) -> Self {
        Self::from_name(name)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0[0].as_ref())?;
        for name in &self.0[1..] {
            write!(f, "/{}", name.as_ref())?;
        }
        Ok(())
    }
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LogError {
    #[error("ID name list must not be empty")]
    InvalidEmptyId,
    #[error(transparent)]
    IncompatibleValue(#[from] Box<IncompatibleValueError>),
}

#[derive(Error, Debug, Clone, PartialEq)]
#[error("\"{id}\": Incompatible value {value:?}, expected {expected}")]
pub struct IncompatibleValueError {
    id: Id,
    value: Loggable,
    expected: String,
}

impl From<IncompatibleValueError> for LogError {
    fn from(value: IncompatibleValueError) -> Self {
        Self::IncompatibleValue(Box::new(value))
    }
}
