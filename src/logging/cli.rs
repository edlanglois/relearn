//! Command-line logger
use super::{Event, LogError, Loggable, Logger};
use enum_map::{enum_map, EnumMap};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::convert::TryInto;
use std::fmt;
use std::ops::Drop;
use std::time::{Duration, Instant};

/// Logger that writes summaries to stderr.
pub struct CLILogger {
    events: EnumMap<Event, EventLog>,

    display_period: Duration,
    last_display_time: Instant,

    average_between_displays: bool,
}

impl CLILogger {
    pub fn new(display_period: Duration, average_between_displays: bool) -> Self {
        CLILogger {
            events: enum_map! { _ => EventLog::new() },
            display_period,
            last_display_time: Instant::now(),
            average_between_displays,
        }
    }

    /// Display the summary and clear all stored data.
    pub fn display(&mut self) {
        println!();
        for (event, mut event_log) in self.events.iter_mut() {
            let summary_size = event_log.index - event_log.summary_start_index;
            if summary_size == 0 {
                continue;
            }

            print!("==== ");
            if self.average_between_displays {
                print!(
                    "{:?}s {} - {}",
                    event,
                    event_log.summary_start_index,
                    event_log.index - 1
                );
            } else {
                print!("{:?} {}", event, event_log.index - 1);
            }
            print!(
                " ({:?} / event)",
                event_log.summary_duration / summary_size.try_into().unwrap()
            );
            println!(" ====");

            for (name, aggregator) in &mut event_log.aggregators {
                println!("{}: {}", name, aggregator);
                aggregator.clear()
            }
            event_log.summary_start_index = event_log.index;
        }
        self.last_display_time = Instant::now();
    }
}

impl Logger for CLILogger {
    fn log<'a>(
        &mut self,
        event: Event,
        name: &'a str,
        value: Loggable,
    ) -> Result<(), LogError<'a>> {
        // The entry API does not currently support lookup with Borrow + IntoOwned
        // Eventually raw_entry can be used but it is not stable yet.
        // For now, make separate get() / insert() calls.
        // The duplicated lookup with insert will only occur once per name since we never remove
        // aggregators.
        let aggregators = &mut self.events[event].aggregators;
        if let Some(aggregator) = aggregators.get_mut(name) {
            if let Err((value, expected)) = aggregator.update(value) {
                return Err(LogError::new(name, value, expected));
            }
        } else {
            let old_value = aggregators.insert(name.into(), Aggregator::new(value));
            assert!(old_value.is_none());
        }
        Ok(())
    }

    fn done(&mut self, event: Event) {
        let event_info = &mut self.events[event];
        event_info.index += 1;

        for aggregator in event_info.aggregators.values_mut() {
            aggregator.commit()
        }

        let time_since_display = self.last_display_time.elapsed();
        event_info.summary_duration = time_since_display;
        if time_since_display < self.display_period {
            return;
        }

        self.display();
    }
}

impl Drop for CLILogger {
    fn drop(&mut self) {
        // Ensure everything is flushed.
        self.display();
    }
}

struct EventLog {
    /// Global index for this event
    index: u64,
    /// Value of `index` at the start of this summary period
    summary_start_index: u64,
    /// Duration of this summary period to the most recent update
    summary_duration: Duration,
    /// An aggregator for each log entry.
    aggregators: BTreeMap<String, Aggregator>,
}

impl EventLog {
    #[allow(clippy::missing_const_for_fn)] // Duration & BTreeMap const new not stabilized
    pub fn new() -> Self {
        Self {
            index: 0,
            summary_start_index: 0,
            summary_duration: Duration::new(0, 0),
            aggregators: BTreeMap::new(),
        }
    }
}

#[derive(Debug)]
enum Aggregator {
    /// Aggregates nothing
    Nothing,
    ScalarMean {
        accumulator: MeanAccumulator,
        pending: Option<<MeanAccumulator as Accumulator>::Prepared>,
    },
    IndexDistribution {
        accumulator: IndexDistributionAccumulator,
        pending: Option<<IndexDistributionAccumulator as Accumulator>::Prepared>,
    },
    MessageCounts {
        accumulator: MessageAccumulator,
        pending: Option<<MessageAccumulator as Accumulator>::Prepared>,
    },
}
use Aggregator::*;

impl Aggregator {
    // future-proofing in case loggable ends up containing non-copy values
    #[allow(clippy::needless_pass_by_value)]
    /// Create a new aggregator from a logged value value.
    fn new(value: Loggable) -> Self {
        match value {
            Loggable::Nothing => Nothing,
            Loggable::Scalar(x) => ScalarMean {
                accumulator: MeanAccumulator::new(),
                pending: Some(x),
            },
            Loggable::IndexSample { value, size } => IndexDistribution {
                accumulator: IndexDistributionAccumulator::new(size),
                pending: Some(value),
            },
            Loggable::Message(message) => MessageCounts {
                accumulator: MessageAccumulator::new(),
                pending: Some(message),
            },
        }
    }

    /// Update an aggregator with a logged value within an event.
    ///
    /// Returns Err((value, expected)) if the value is incompatible with this aggregator.
    fn update(&mut self, value: Loggable) -> Result<(), (Loggable, String)> {
        match self {
            Nothing => match value {
                Loggable::Nothing => {}
                _ => return Err((value, "Nothing".into())),
            },
            ScalarMean {
                accumulator,
                pending,
            } => *pending = Some(accumulator.prepare(value)?),
            IndexDistribution {
                accumulator,
                pending,
            } => *pending = Some(accumulator.prepare(value)?),
            MessageCounts {
                accumulator,
                pending,
            } => *pending = Some(accumulator.prepare(value)?),
        };
        Ok(())
    }

    /// Commit the pending values into the aggregate.
    fn commit(&mut self) {
        match self {
            Nothing => {}
            ScalarMean {
                accumulator,
                pending,
            } => {
                if let Some(value) = pending.take() {
                    accumulator.insert(value)
                }
            }
            IndexDistribution {
                accumulator,
                pending,
            } => {
                if let Some(value) = pending.take() {
                    accumulator.insert(value)
                }
            }
            MessageCounts {
                accumulator,
                pending,
            } => {
                if let Some(value) = pending.take() {
                    accumulator.insert(value)
                }
            }
        }
    }

    /// Clear the aggregated values (but not the pending values)
    fn clear(&mut self) {
        match self {
            Nothing => {}
            ScalarMean {
                accumulator,
                pending: _,
            } => {
                accumulator.clear();
            }
            IndexDistribution {
                accumulator,
                pending: _,
            } => {
                accumulator.clear();
            }
            MessageCounts {
                accumulator,
                pending: _,
            } => {
                accumulator.clear();
            }
        }
    }
}

/// Display the commited aggregated value.
impl fmt::Display for Aggregator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Nothing => write!(f, "Nothing"),
            ScalarMean {
                accumulator,
                pending: _,
            } => accumulator.fmt(f),
            IndexDistribution {
                accumulator,
                pending: _,
            } => accumulator.fmt(f),
            MessageCounts {
                accumulator,
                pending: _,
            } => accumulator.fmt(f),
        }
    }
}

/// Accumulate statistics of a loggable.
trait Accumulator: 'static + fmt::Display {
    /// Type for prepared values.
    type Prepared;

    /// Prepare a value to be inserted into the accumulator.
    ///
    /// Used when a value is logged during an event.
    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)>;

    /// Insert a new prepared value into the accumulation.
    ///
    /// Used at the end of an event.
    fn insert(&mut self, value: Self::Prepared);

    /// Clear the accumulated values.
    fn clear(&mut self);
}

#[derive(Debug)]
struct MeanAccumulator {
    sum: f64,
    count: u64,
}

impl MeanAccumulator {
    pub const fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
}

impl Accumulator for MeanAccumulator {
    type Prepared = f64;

    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)> {
        if let Loggable::Scalar(x) = value {
            Ok(x)
        } else {
            Err((value, "Scalar".into()))
        }
    }

    fn insert(&mut self, value: Self::Prepared) {
        self.sum += value;
        self.count += 1;
    }

    fn clear(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

impl fmt::Display for MeanAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.sum / (self.count as f64))
    }
}

#[derive(Debug)]
struct IndexDistributionAccumulator {
    counts: Vec<u64>,
}

impl IndexDistributionAccumulator {
    fn new(size: usize) -> Self {
        Self {
            counts: vec![0; size],
        }
    }
}

impl Accumulator for IndexDistributionAccumulator {
    type Prepared = usize;

    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)> {
        match value {
            Loggable::IndexSample { value, size } if self.counts.len() == size => Ok(value),
            v => Err((v, format!("IndexSample{{size: {}}}", self.counts.len()))),
        }
    }

    fn insert(&mut self, value: Self::Prepared) {
        self.counts[value] += 1;
    }

    fn clear(&mut self) {
        for count in &mut self.counts {
            *count = 0;
        }
    }
}

impl fmt::Display for IndexDistributionAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let total: u64 = self.counts.iter().sum();
        if total > 0 {
            write!(f, "[")?;
            let mut first = true;
            for c in &self.counts {
                if first {
                    first = false;
                } else {
                    write!(f, ", ")?;
                }
                write!(f, "{:.3}", (*c as f64) / (total as f64))?;
            }
            write!(f, "]")
        } else {
            write!(f, "None")
        }
    }
}

#[derive(Debug)]
struct MessageAccumulator {
    /// Number of occurrences of each unique message.
    message_counts: BTreeMap<Cow<'static, str>, usize>,
}

impl MessageAccumulator {
    pub fn new() -> Self {
        Self {
            message_counts: BTreeMap::new(),
        }
    }
}

impl Accumulator for MessageAccumulator {
    type Prepared = Cow<'static, str>;

    fn prepare(&self, value: Loggable) -> Result<Self::Prepared, (Loggable, String)> {
        if let Loggable::Message(message) = value {
            Ok(message)
        } else {
            Err((value, "Message".into()))
        }
    }

    fn insert(&mut self, message: Self::Prepared) {
        *self.message_counts.entry(message).or_insert(0) += 1;
    }

    fn clear(&mut self) {
        self.message_counts.clear();
    }
}

impl fmt::Display for MessageAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.message_counts.len() == 1 {
            for (message, count) in &self.message_counts {
                write!(f, "[x{}] {}", count, message)?;
            }
        } else {
            for (message, count) in &self.message_counts {
                write!(f, "\n\t[x{}] {}", count, message)?;
            }
        }
        Ok(())
    }
}
