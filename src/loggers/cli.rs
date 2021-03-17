use super::{Event, LogError, Loggable, Logger};
use enum_map::{enum_map, EnumMap};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
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
        sum: f64,
        count: u64,
        pending_sum: f64,
        pending_count: u64,
    },
    IndexDistribution {
        counts: Vec<u64>,
        pending_counts: Vec<u64>,
    },
}
use Aggregator::*;
use Loggable::*;

impl Aggregator {
    /// Create a new aggregator from a logged value value.
    fn new(value: Loggable) -> Self {
        match value {
            Scalar(x) => ScalarMean {
                sum: 0.0,
                count: 0,
                pending_sum: x,
                pending_count: 1,
            },
            IndexSample { value, size } => {
                let mut pending_counts = vec![0; size];
                pending_counts[value] = 1;
                IndexDistribution {
                    counts: vec![0; size],
                    pending_counts,
                }
            }
        }
    }

    /// Update an aggregator with a logged value within an event.
    ///
    /// Returns Err((value, expected)) if the value is incompatible with this aggregator.
    fn update(&mut self, value: Loggable) -> Result<(), (Loggable, String)> {
        match (self, value) {
            (
                ScalarMean {
                    sum: _,
                    count: _,
                    pending_sum,
                    pending_count,
                },
                Scalar(x),
            ) => {
                *pending_sum += x;
                *pending_count += 1;
            }

            (
                IndexDistribution {
                    counts: _,
                    pending_counts,
                },
                IndexSample { value, size },
            ) if pending_counts.len() == size => {
                pending_counts[value] += 1;
            }

            (s, value) => return Err((value, s.expected())),
        }
        Ok(())
    }

    /// Commit the pending values into the aggregate.
    fn commit(&mut self) {
        match self {
            ScalarMean {
                sum,
                count,
                pending_sum,
                pending_count,
            } => {
                *sum += *pending_sum;
                *count += *pending_count;
                *pending_sum = 0.0;
                *pending_count = 0;
            }

            IndexDistribution {
                counts,
                pending_counts,
            } => {
                for (c, pc) in counts.iter_mut().zip(pending_counts.iter_mut()) {
                    *c = *pc;
                    *pc = 0;
                }
            }
        }
    }

    /// Clear the aggregated values (but not the pending values)
    fn clear(&mut self) {
        match self {
            ScalarMean {
                sum,
                count,
                pending_sum: _,
                pending_count: _,
            } => {
                *sum = 0.0;
                *count = 0;
            }

            IndexDistribution {
                counts,
                pending_counts: _,
            } => {
                for c in counts.iter_mut() {
                    *c = 0;
                }
            }
        }
    }

    /// A string describing the expected loggable type
    fn expected(&self) -> String {
        match self {
            ScalarMean {
                sum: _,
                count: _,
                pending_sum: _,
                pending_count: _,
            } => "Scalar".into(),
            IndexDistribution {
                counts,
                pending_counts: _,
            } => format!("IndexSample{{size: {}}}", counts.len()),
        }
    }
}

/// Display the commited aggregated value.
impl fmt::Display for Aggregator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ScalarMean {
                sum,
                count,
                pending_sum: _,
                pending_count: _,
            } => {
                write!(f, "{}", sum / (*count as f64))
            }
            IndexDistribution {
                counts,
                pending_counts: _,
            } => {
                let total: u64 = counts.iter().sum();
                if total > 0 {
                    write!(f, "[")?;
                    let mut first = true;
                    for c in counts.iter() {
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
    }
}
