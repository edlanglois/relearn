use super::{Event, LogError, Loggable, Logger};
use enum_map::{enum_map, EnumMap};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;
use std::ops::{AddAssign, Drop};
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
        println!("");
        for (event, mut event_log) in self.events.iter_mut() {
            let summary_delta = event_log.index - event_log.summary_start_index;
            if summary_delta == 0 {
                continue;
            }
            let summary_size = summary_delta - 1;

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
                event_log.summary_duration / (summary_size as u32)
            );
            println!(" ====");

            for (name, aggregator) in event_log.aggregators.iter_mut() {
                println!("{}: {}", name, aggregator);
                aggregator.clear()
            }
            event_log.summary_start_index = event_log.index;
        }
        self.last_display_time = Instant::now();
    }
}

impl Logger for CLILogger {
    fn log(&mut self, event: Event, name: &'static str, value: Loggable) -> Result<(), LogError> {
        match self.events[event].aggregators.entry(name) {
            Entry::Vacant(v) => {
                v.insert(Aggregator::new(value));
            }
            Entry::Occupied(o) => {
                if let Err((value, expected)) = o.into_mut().update(value) {
                    return Err(LogError::new(name, value, expected));
                }
            }
        }
        Ok(())
    }

    fn done(&mut self, event: Event) {
        let event_info = &mut self.events[event];
        event_info.index += 1;

        for (_, aggregator) in event_info.aggregators.iter_mut() {
            aggregator.commit()
        }

        let time_since_display = self.last_display_time.elapsed();
        event_info.summary_duration = time_since_display;
        if time_since_display < self.display_period {
            return;
        }

        // Don't output after steps - prefer complete episodes or epochs
        if let Event::Step = event {
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
    aggregators: BTreeMap<&'static str, Aggregator>,
}

impl EventLog {
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
    ScalarMean {
        accumulated: MeanAccumulator,
        pending: MeanAccumulator,
    },
    IndexDistribution {
        accumulated: IndexDistributionAccumulator,
        pending: IndexDistributionAccumulator,
    },
}
use Aggregator::*;
use Loggable::*;

impl Aggregator {
    /// Create a new aggregator from a logged value value.
    fn new(value: Loggable) -> Self {
        match value {
            Scalar(x) => ScalarMean {
                accumulated: MeanAccumulator::new(),
                pending: x.into(),
            },
            IndexSample { value, size } => {
                let mut pending = IndexDistributionAccumulator::new(size);
                pending.counts[value] += 1;
                IndexDistribution {
                    accumulated: IndexDistributionAccumulator::new(size),
                    pending,
                }
            }
        }
    }

    /// Update an aggregator with a logged value within an event.
    ///
    /// Returns Err((value, expected)) if the value is incompatible with this aggregator.
    fn update(&mut self, value: Loggable) -> Result<(), (Loggable, String)> {
        match self {
            ScalarMean {
                accumulated: _,
                pending,
            } => pending.update(value),
            IndexDistribution {
                accumulated: _,
                pending,
            } => pending.update(value),
        }
    }

    /// Commit the pending values into the aggregate.
    fn commit(&mut self) {
        match self {
            ScalarMean {
                accumulated,
                pending,
            } => {
                *accumulated += pending;
                pending.reset()
            }
            IndexDistribution {
                accumulated,
                pending,
            } => {
                *accumulated += pending;
                pending.reset()
            }
        }
    }

    /// Clear the aggregated values (but not the pending values)
    fn clear(&mut self) {
        match self {
            ScalarMean {
                accumulated,
                pending: _,
            } => {
                accumulated.reset();
            }
            IndexDistribution {
                accumulated,
                pending: _,
            } => {
                accumulated.reset();
            }
        }
    }
}

/// Display the commited aggregated value.
impl fmt::Display for Aggregator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ScalarMean {
                accumulated,
                pending: _,
            } => accumulated.fmt(f),
            IndexDistribution {
                accumulated,
                pending: _,
            } => accumulated.fmt(f),
        }
    }
}

/// Accumulate statistics of a loggable.
trait Accumulator:
    'static
    + TryFrom<Loggable, Error = (Loggable, &'static str)>
    + AddAssign<&'static Self>
    + fmt::Display
{
    /// Add a new loggable value to the accumulator.
    ///
    /// Used when a value is logged multiple times in an event.
    fn update(&mut self, value: Loggable) -> Result<(), (Loggable, String)>;

    /// Reset the accumulator to empty
    fn reset(&mut self);
}

#[derive(Debug)]
struct MeanAccumulator {
    sum: f64,
    count: u64,
}

impl MeanAccumulator {
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
}

impl From<f64> for MeanAccumulator {
    fn from(value: f64) -> Self {
        Self {
            sum: value,
            count: 1,
        }
    }
}

impl Accumulator for MeanAccumulator {
    fn update(&mut self, value: Loggable) -> Result<(), (Loggable, String)> {
        if let Scalar(x) = value {
            self.sum += x;
            self.count += 1;
            Ok(())
        } else {
            Err((value, "Scalar".into()))
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

impl TryFrom<Loggable> for MeanAccumulator {
    type Error = (Loggable, &'static str);

    fn try_from(value: Loggable) -> Result<Self, Self::Error> {
        if let Scalar(x) = value {
            Ok(x.into())
        } else {
            Err((value, "Scalar"))
        }
    }
}

impl AddAssign<&Self> for MeanAccumulator {
    fn add_assign(&mut self, other: &Self) {
        self.sum += other.sum;
        self.count += other.count;
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
    fn update(&mut self, value: Loggable) -> Result<(), (Loggable, String)> {
        match value {
            IndexSample { value, size } if self.counts.len() == size => {
                self.counts[value] += 1;
                Ok(())
            }
            v => Err((v, format!("IndexSample{{size: {}}}", self.counts.len()))),
        }
    }

    fn reset(&mut self) {
        for count in self.counts.iter_mut() {
            *count = 0;
        }
    }
}

impl TryFrom<Loggable> for IndexDistributionAccumulator {
    type Error = (Loggable, &'static str);

    fn try_from(value: Loggable) -> Result<Self, Self::Error> {
        if let IndexSample { value, size } = value {
            let mut acc = Self::new(size);
            acc.counts[value] += 1;
            Ok(acc)
        } else {
            Err((value, "IndexSample"))
        }
    }
}

impl AddAssign<&Self> for IndexDistributionAccumulator {
    fn add_assign(&mut self, other: &Self) {
        for (count, other_count) in self.counts.iter_mut().zip(other.counts.iter()) {
            *count += other_count;
        }
    }
}

impl fmt::Display for IndexDistributionAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let total: u64 = self.counts.iter().sum();
        if total > 0 {
            write!(f, "[")?;
            let mut first = true;
            for c in self.counts.iter() {
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
