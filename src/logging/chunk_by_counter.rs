use super::chunk::{ChunkSummary, Chunker, Flush, Node};
use super::{Id, Loggable};
use std::collections::BTreeMap;

/// Chunk summaries at fixed multiples of a counter (for [`ChunkLogger`][super::ChunkLogger]).
///
/// Flushes the summary after the counter update has been included in the summary.
/// As such, users should log counter increments last, after other logs associated with that
/// counter. This ensures that the logs will be
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByCounter {
    /// ID of the counter to use
    pub counter: Id,
    /// Chunk length in terms of the counter.
    pub interval: u64,
}

impl ByCounter {
    pub const fn new(counter: Id, interval: u64) -> Self {
        Self { counter, interval }
    }

    pub fn of_path<T: IntoIterator<Item = &'static str>>(path: T, interval: u64) -> Self {
        Self {
            counter: Id::from_iter(path),
            interval,
        }
    }
}

impl Chunker for ByCounter {
    /// Whether the ID matches counter
    type Context = bool;

    fn flush_pre_log(&self, _: &BTreeMap<Id, Node>, id: &Id, _: &Loggable) -> Flush<Self::Context> {
        Flush::MaybePostLog(&self.counter == id)
    }

    fn flush_post_log(&self, is_counter: Self::Context, updated: &ChunkSummary) -> bool {
        if !is_counter {
            return false;
        }
        if let ChunkSummary::Counter {
            increment,
            initial_value,
        } = updated
        {
            // The increment is > 0 in most cases anyway so displaying on 0 shouldn't be a big risk
            (increment + initial_value) % self.interval == 0
        } else {
            panic!("Target ID {} is not a counter", self.counter);
        }
    }

    fn flushed(&mut self) {}
}
