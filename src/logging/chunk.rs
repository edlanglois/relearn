use super::{Id, LogError, Loggable, StatsLogger};
use crate::utils::stats::OnlineMeanVariance;
use coarsetime::{Duration as CDuration, Instant as CInstant};
use std::borrow::Cow;
use std::collections::{btree_map::Entry, BTreeMap};
use std::ops::Drop;
use std::time::{Duration, Instant};

/// Logs time series statistics by breaking the time series into chunks and summarizing each chunk.
pub struct ChunkLogger<B: LoggerBackend> {
    // Coarse time is used because the current time is checked on every log event,
    // which might be quite frequent for per-step values so the time checks should be fast.
    // The accuracy of the clock (~1ms for coarse vs. ~1ns for regular) is not very important
    // since it is only used for checking whether the chunk duration has elapsed (~1s).
    chunk_duration: CDuration,
    coarse_chunk_start: CInstant,
    // Precise chunk start time for use when recording statistics.
    chunk_start: Instant,

    // A binary tree is used so that keys are retrieved in sorted order
    summaries: BTreeMap<Id, Node>,

    backend: B,
}

impl<B: LoggerBackend> ChunkLogger<B> {
    /// Create a new logger with the given backend.
    pub fn from_backend(chunk_duration: Duration, backend: B) -> Self {
        Self {
            chunk_duration: CDuration::new(chunk_duration.as_secs(), chunk_duration.subsec_nanos()),
            coarse_chunk_start: CInstant::now(),
            chunk_start: Instant::now(),
            summaries: BTreeMap::new(),
            backend,
        }
    }
}

impl<B: Default + LoggerBackend> Default for ChunkLogger<B> {
    fn default() -> Self {
        Self::from_backend(Duration::from_secs(1), B::default())
    }
}

impl<B: LoggerBackend> StatsLogger for ChunkLogger<B> {
    fn log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        // Check whether the chunk duration has elapsed.
        // This is done before logging because logs are likely to occur in bursts with the duration
        // elapsing in between. Logging first would cause the first value of the burst to be split
        // into a separate chunk from the rest.
        if self.coarse_chunk_start.elapsed() > self.chunk_duration {
            self.flush()
        }

        self.log_no_flush(id, value)
    }

    fn log_no_flush(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        match self.summaries.entry(id) {
            Entry::Vacant(e) => {
                e.insert(Node::new(value.into()));
            }
            Entry::Occupied(mut e) => {
                e.get_mut().push(value)?;
            }
        };
        Ok(())
    }

    fn flush(&mut self) {
        self.backend.record_summaries(
            self.summaries.iter().filter_map(|(id, node)| {
                if node.dirty {
                    Some((id, &node.summary))
                } else {
                    None
                }
            }),
            self.chunk_start.elapsed(),
        );

        // Reset
        for node in self.summaries.values_mut() {
            node.reset();
        }
        self.coarse_chunk_start = CInstant::now();
        self.chunk_start = Instant::now();
    }
}

/// Flush when dropped
impl<B: LoggerBackend> Drop for ChunkLogger<B> {
    fn drop(&mut self) {
        self.flush();
    }
}

pub trait LoggerBackend: Send {
    /// Record the given summaries to the backend.
    fn record_summaries<'a, I>(&mut self, summaries: I, elapsed: Duration)
    where
        I: Iterator<Item = (&'a Id, &'a ChunkSummary)>;
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    /// Variable chunk summary
    summary: ChunkSummary,
    /// Whether the summary has been updated in this chunk
    dirty: bool,
}

impl Node {
    pub const fn new(summary: ChunkSummary) -> Self {
        Self {
            summary,
            dirty: true,
        }
    }

    fn push(&mut self, value: Loggable) -> Result<(), LogError> {
        self.dirty = true;
        self.summary.push(value)
    }

    fn reset(&mut self) {
        self.dirty = false;
        self.summary.reset()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct CounterSummary {
    value: u64,
    initial_value: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkSummary {
    Nothing,
    Counter {
        increment: u64,
        initial_value: u64,
    },
    Duration {
        stats: OnlineMeanVariance<f64>,
    },
    Scalar {
        stats: OnlineMeanVariance<f64>,
    },
    Index {
        counts: Vec<usize>,
    },
    Message {
        counts: BTreeMap<Cow<'static, str>, usize>,
    },
}

impl From<Loggable> for ChunkSummary {
    fn from(value: Loggable) -> Self {
        match value {
            Loggable::Nothing => Self::Nothing,
            Loggable::CounterIncrement(i) => Self::Counter {
                increment: i,
                initial_value: 0,
            },
            Loggable::Duration(d) => {
                let mut stats = OnlineMeanVariance::new();
                stats.push(d.as_secs_f64());
                Self::Duration { stats }
            }
            Loggable::Scalar(v) => {
                let mut stats = OnlineMeanVariance::new();
                stats.push(v);
                Self::Scalar { stats }
            }
            Loggable::Index { value: v, size } => {
                let mut counts = vec![0; size];
                counts[v] += 1;
                Self::Index { counts }
            }
            Loggable::Message(s) => {
                let mut counts = BTreeMap::new();
                let prev = counts.insert(s, 1);
                assert!(prev.is_none());
                Self::Message { counts }
            }
        }
    }
}

impl ChunkSummary {
    /// Add a value to the summary
    ///
    /// Returns and error and does not insert the value if it is incompatible with the current
    /// summary. The value will be incompatible if the summary was created from a different
    /// loggable variant, or if some other structure of the loggable is different.
    fn push(&mut self, value: Loggable) -> Result<(), LogError> {
        match (self, value) {
            (Self::Nothing, Loggable::Nothing) => {}
            (
                Self::Counter {
                    increment,
                    initial_value: _,
                },
                Loggable::CounterIncrement(i),
            ) => {
                *increment += i;
            }
            (Self::Duration { stats }, Loggable::Duration(d)) => {
                stats.push(d.as_secs_f64());
            }
            (Self::Scalar { stats }, Loggable::Scalar(v)) => stats.push(v),
            (Self::Index { counts }, Loggable::Index { value: v, size }) => {
                if counts.len() != size {
                    return Err(LogError::IncompatibleIndexSize {
                        prev: counts.len(),
                        now: size,
                    });
                }
                counts[v] += 1;
            }
            (Self::Message { counts }, Loggable::Message(s)) => {
                *counts.entry(s).or_insert(0) += 1;
            }
            (summary, value) => {
                return Err(LogError::IncompatibleValue {
                    prev: summary.loggable_variant_name(),
                    now: value.variant_name(),
                })
            }
        };
        Ok(())
    }

    /// Reset for the start of the next chunk.
    fn reset(&mut self) {
        match self {
            Self::Nothing => {}
            Self::Counter {
                increment,
                initial_value,
            } => {
                *initial_value += *increment;
                *increment = 0
            }
            Self::Duration { stats } => *stats = OnlineMeanVariance::new(),
            Self::Scalar { stats } => *stats = OnlineMeanVariance::new(),
            Self::Index { counts } => counts.iter_mut().for_each(|c| *c = 0),
            Self::Message { counts } => counts.clear(),
        }
    }

    /// The name of the associated loggable variant
    const fn loggable_variant_name(&self) -> &'static str {
        match self {
            Self::Nothing => "Nothing",
            Self::Counter {
                increment: _,
                initial_value: _,
            } => "CounterIncrement",
            Self::Duration { stats: _ } => "Duration",
            Self::Scalar { stats: _ } => "Scalar",
            Self::Index { counts: _ } => "Index",
            Self::Message { counts: _ } => "Message",
        }
    }
}
