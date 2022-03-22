use super::{Id, LogError, Loggable, StatsLogger};
use crate::utils::stats::OnlineMeanVariance;
use std::borrow::Cow;
use std::collections::{btree_map::Entry, BTreeMap};
use std::ops::Drop;
use std::time::{Duration, Instant};

/// Control the aggregation of logs into summaries and summaries into chunks.
pub trait Chunker: Send {
    /// Saved context for deciding whether to log post-flush
    ///
    /// # Design Note
    /// This complexity is necesary because Rust does not provide a way to get a reference to the
    /// key after insertion so `flush_post_log` would not otherwise be able to depend on the value
    /// of `Id` (without cloning Id).
    /// See <https://stackoverflow.com/questions/32401857>
    type Context;

    /// Decide whether to flush the current chunk before the given value is added to the summary.
    fn flush_pre_log(
        &self,
        summaries: &BTreeMap<Id, Node>,
        id: &Id,
        value: &Loggable,
    ) -> Flush<Self::Context>;

    /// Decide whether to flush the current chunk after the value has been added to the summary.
    fn flush_post_log(&self, context: Self::Context, updated: &ChunkSummary) -> bool;

    /// Indicate that the current chunk has been flushed
    fn flushed(&mut self);
}

/// When and whether to flush the current chunk.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Flush<T> {
    /// Do not flush
    No,
    /// Flush before logging
    PreLog,
    /// Maybe flush after logging, determined by [`Chunker::flush_post_log`].
    MaybePostLog(T),
}

/// Write out summaries to a backend.
pub trait SummaryWriter: Send {
    fn write_summaries<'a, I>(&mut self, summaries: I, elapsed: Duration)
    where
        I: Iterator<Item = (&'a Id, &'a ChunkSummary)>;
}

/// Logs time series statistics by breaking the time series into chunks and summarizing each chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkLogger<C: Chunker, W: SummaryWriter> {
    chunker: C,
    writer: W,

    // A binary tree is used so that keys are retrieved in sorted order
    summaries: BTreeMap<Id, Node>,

    // Start time of the current chunk.
    //
    // Passed to writers, used for measuring event frequences for example.
    chunk_start: Instant,
}

impl<C: Chunker, W: SummaryWriter> ChunkLogger<C, W> {
    pub fn new(chunker: C, writer: W) -> Self {
        Self {
            chunker,
            writer,
            summaries: BTreeMap::new(),
            chunk_start: Instant::now(),
        }
    }
}

impl<C: Chunker + Default, W: SummaryWriter + Default> Default for ChunkLogger<C, W> {
    fn default() -> Self {
        Self::new(C::default(), W::default())
    }
}

impl<C: Chunker, W: SummaryWriter> StatsLogger for ChunkLogger<C, W> {
    fn log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        let flush_request = self.chunker.flush_pre_log(&self.summaries, &id, &value);
        if matches!(flush_request, Flush::PreLog) {
            self.flush();
        }

        let node = match self.summaries.entry(id) {
            Entry::Vacant(e) => e.insert(Node::new(value.into())),
            Entry::Occupied(e) => {
                let node = e.into_mut();
                node.push(value)?;
                node
            }
        };

        if let Flush::MaybePostLog(context) = flush_request {
            if self.chunker.flush_post_log(context, &node.summary) {
                self.flush()
            }
        }
        Ok(())
    }

    fn log_no_flush(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        match self.summaries.entry(id) {
            Entry::Vacant(e) => {
                e.insert(Node::new(value.into()));
            }
            Entry::Occupied(e) => {
                e.into_mut().push(value)?;
            }
        };
        Ok(())
    }

    fn flush(&mut self) {
        self.writer.write_summaries(
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
        self.chunk_start = Instant::now();
        self.chunker.flushed();
    }
}

/// Flush when dropped
impl<C: Chunker, W: SummaryWriter> Drop for ChunkLogger<C, W> {
    fn drop(&mut self) {
        self.flush();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
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
