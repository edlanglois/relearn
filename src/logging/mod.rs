//! Logging statistics from simulation runs
pub mod chunk;
mod display;
mod tensorboard;

pub use chunk::ChunkLogger;
pub use display::{DisplayLogger, DisplayLoggerConfig};
pub use tensorboard::TensorBoardLogger;

use smallvec::SmallVec;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::iter;
use std::slice;
use std::time::Duration;
use thiserror::Error;

/// Log a time series of statistics.
///
/// Statistics with the same name may be aggregated or summarized over some time period.
pub trait StatsLogger: Send {
    /// Log a value associated with an ID.
    ///
    /// # Args
    /// * `id` -
    ///     Unique identifier of the statistic to log. Used to track the value over time.
    ///     It is an error to use the same identifier with values that have different
    ///     [`Loggable`] variants or are otherwise structurally incompatible.
    ///
    ///     The id can be created from a string with `into()`.
    ///     It is recommended that users pass ids containing the name only, not a namespace.
    ///     Namespaces should be managed with [`StatsLogger::with_scope`].
    ///
    ///     The namespace on the id is used to internally track scope and, unexpectedly
    ///     for the external interface, it is an **outer** namespace.
    ///     Loggers may append inner namespaces.
    ///
    /// * `value` - The value to log.
    fn log(&mut self, id: Id, value: Loggable) -> Result<(), LogError>;

    /// Log a value without checking whether the logger should be flushed.
    ///
    /// Useful in a sequence of logs or in high frequency logs to avoid the cost of checking each
    /// time.
    /// See [`StatsLogger::log`] for more documentation.
    fn log_no_flush(&mut self, id: Id, value: Loggable) -> Result<(), LogError>;

    /// Record any remaining data in the logger that has not yet been recorded.
    fn flush(&mut self);

    /// Wrap this logger such that an inner scope is added to all logged ids.
    ///
    /// This can be called on a reference for a temporary scope: `(&mut logger).with_scope(...)`
    fn with_scope(self, scope: &'static str) -> ScopedLogger<Self>
    where
        Self: Sized,
    {
        ScopedLogger::new(scope, self)
    }

    // Convenience functions

    /// Log an increment to a named counter (convenience function).
    ///
    /// Panics if this name was previously used to log a value of a different type.
    #[inline]
    fn log_counter_increment(&mut self, name: &'static str, increment: u64) {
        self.log(name.into(), Loggable::CounterIncrement(increment))
            .unwrap()
    }

    /// Log a named duration (convenience function).
    ///
    /// Panics if this name was previously used to log a value of a different type.
    #[inline]
    fn log_duration(&mut self, name: &'static str, duration: Duration) {
        self.log(name.into(), Loggable::Duration(duration)).unwrap()
    }

    /// Log a named scalar value (convenience function).
    ///
    /// Panics if this name was previously used to log a value of a different type.
    #[inline]
    fn log_scalar(&mut self, name: &'static str, value: f64) {
        self.log(name.into(), Loggable::Scalar(value)).unwrap()
    }

    /// Log a named index in `0` to `size - 1` (convenience function).
    ///
    /// Panics if this name was previously used to log a value of a different type
    /// or an index value with a different size.
    #[inline]
    fn log_index(&mut self, name: &'static str, value: usize, size: usize) {
        self.log(name.into(), Loggable::Index { value, size })
            .unwrap()
    }

    /// Log a named message (convenience function).
    ///
    /// Panics if this name was previously used to log a value of a different type.
    #[inline]
    fn log_message(&mut self, name: &'static str, message: &'static str) {
        self.log(name.into(), Loggable::Message(message.into()))
            .unwrap()
    }
}

/// Implement `StatsLogger` for a deref-able wrapper type generic over `T: StatsLogger + ?Sized`.
macro_rules! impl_wrapped_stats_logger {
    ($wrapper:ty) => {
        impl<T> StatsLogger for $wrapper
        where
            T: StatsLogger + ?Sized,
        {
            #[inline]
            fn log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
                T::log(self, id, value)
            }

            #[inline]
            fn log_no_flush(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
                T::log_no_flush(self, id, value)
            }

            #[inline]
            fn flush(&mut self) {
                T::flush(self)
            }
        }
    };
}
impl_wrapped_stats_logger!(&'_ mut T);
impl_wrapped_stats_logger!(Box<T>);

/// Value that can be logged.
///
/// # Design Note
/// This is an enum to simplify the logging interface.
/// The options are:
/// * multiple methods
///     - pro: no dynamic dispatch; supports trait objects
///     - con: duplicates similar functionality; cannot batch log different types
/// * enum
///     - pro: simple interface with few methods; supports trait objects; batch log `log_items`
///     - con: dynamic dispatch, can fail with errors; wasted space in loggable / summary
/// * traits
///     - pro: possibly less dynamic dispatch in some cases
///     - con: must downcast for backend; complex interface; hard to do trait objects
#[derive(Debug, Clone, PartialEq)]
pub enum Loggable {
    Nothing,
    CounterIncrement(u64),
    Duration(Duration),
    Scalar(f64),
    Index { value: usize, size: usize },
    Message(Cow<'static, str>),
}

impl From<f64> for Loggable {
    fn from(scalar: f64) -> Self {
        Self::Scalar(scalar)
    }
}

impl From<Duration> for Loggable {
    fn from(duration: Duration) -> Self {
        Self::Duration(duration)
    }
}

impl From<&'static str> for Loggable {
    fn from(s: &'static str) -> Self {
        Self::Message(s.into())
    }
}

impl From<String> for Loggable {
    fn from(s: String) -> Self {
        Self::Message(s.into())
    }
}

impl Loggable {
    const fn variant_name(&self) -> &'static str {
        use Loggable::*;
        match self {
            Nothing => "Nothing",
            CounterIncrement(_) => "CounterIncrement",
            Duration(_) => "Duration",
            Scalar(_) => "Scalar",
            Index { value: _, size: _ } => "Index",
            Message(_) => "Message",
        }
    }
}

/// A hierarchical identifier.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Id {
    /// Base (innermost) name of the identifier.
    name: Cow<'static, str>,
    /// Hierarchical namespace in reverse order from innermost to outermost (top-level)
    namespace: SmallVec<[&'static str; 6]>,
}

impl PartialOrd for Id {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Id {
    fn cmp(&self, other: &Self) -> Ordering {
        self.components().cmp(other.components())
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let total_len = self.name.len() + self.namespace.iter().map(|s| s.len() + 1).sum::<usize>();

        // Pad on the left for right alignment if necessary
        if let Some(width) = f.width() {
            if width > total_len && matches!(f.align(), Some(fmt::Alignment::Right)) {
                let c = f.fill();
                for _ in 0..(width - total_len) {
                    write!(f, "{}", c)?;
                }
            }
        }

        for scope in self.namespace.iter().rev() {
            write!(f, "{}/", scope)?;
        }
        write!(f, "{}", self.name)?;

        // Pad on the right for left alignment if necessary
        if let Some(width) = f.width() {
            if width > total_len && matches!(f.align(), Some(fmt::Alignment::Left)) {
                let c = f.fill();
                for _ in 0..(width - total_len) {
                    write!(f, "{}", c)?;
                }
            }
        }

        Ok(())
    }
}

impl<T> From<T> for Id
where
    T: Into<Cow<'static, str>>,
{
    #[inline]
    fn from(name: T) -> Self {
        let name = name.into();
        debug_assert!(
            !name.contains('/'),
            "path separators are not allowed in Id name; \
            use [...].collect() or logger.with_scope(...) instead"
        );
        Self {
            name,
            namespace: SmallVec::new(),
        }
    }
}

impl Id {
    /// Add a new inner scope to the namespace.
    fn with_inner_scope(mut self, scope: &'static str) -> Self {
        self.namespace.push(scope);
        self
    }

    /// Iterator over ID components
    pub fn components(
        &self,
    ) -> iter::Chain<iter::Cloned<iter::Rev<slice::Iter<&str>>>, iter::Once<&str>> {
        self.namespace
            .iter()
            .rev()
            .cloned() // &&str -> &str
            .chain(iter::once(self.name.as_ref()))
    }
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LogError {
    #[error("incompatible value type; previously {prev} now {now}")]
    IncompatibleValue {
        prev: &'static str,
        now: &'static str,
    },
    #[error("incompatible index size; previously {prev} now {now}")]
    IncompatibleIndexSize { prev: usize, now: usize },
}

/// No-op logger
impl StatsLogger for () {
    fn log(&mut self, _: Id, _: Loggable) -> Result<(), LogError> {
        Ok(())
    }

    fn log_no_flush(&mut self, _: Id, _: Loggable) -> Result<(), LogError> {
        Ok(())
    }

    fn flush(&mut self) {}
}

/// Pair of loggers; logs to both.
impl<A, B> StatsLogger for (A, B)
where
    A: StatsLogger,
    B: StatsLogger,
{
    fn log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        // Log to both even if one fails
        let r1 = self.0.log(id.clone(), value.clone());
        let r2 = self.1.log(id, value);
        r1.and(r2)
    }

    fn log_no_flush(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        // Log to both even if one fails
        let r1 = self.0.log_no_flush(id.clone(), value.clone());
        let r2 = self.1.log_no_flush(id, value);
        r1.and(r2)
    }

    fn flush(&mut self) {
        self.0.flush();
        self.1.flush();
    }
}

/// Wraps all logged names with a scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScopedLogger<L> {
    scope: &'static str,
    logger: L,
}

impl<L> ScopedLogger<L> {
    #[inline]
    pub const fn new(scope: &'static str, logger: L) -> Self {
        Self { scope, logger }
    }
}

impl<L: StatsLogger> StatsLogger for ScopedLogger<L> {
    #[inline]
    fn log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        self.logger.log(id.with_inner_scope(self.scope), value)
    }

    #[inline]
    fn log_no_flush(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        self.logger
            .log_no_flush(id.with_inner_scope(self.scope), value)
    }

    #[inline]
    fn flush(&mut self) {
        self.logger.flush()
    }
}
