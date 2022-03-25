use super::chunk::{ChunkSummary, Chunker};
use super::{Id, Loggable};

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
    state: State,
}

impl ByCounter {
    pub const fn new(counter: Id, interval: u64) -> Self {
        Self {
            counter,
            interval,
            state: State::NoFlush,
        }
    }

    pub fn of_path<T: IntoIterator<Item = &'static str>>(path: T, interval: u64) -> Self {
        Self::new(Id::from_iter(path), interval)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum State {
    NoFlush,
    IdMatch,
    Flush,
}

impl Chunker for ByCounter {
    #[inline]
    fn note_log(&mut self, id: &Id, _: &Loggable) {
        debug_assert!(
            matches!(self.state, State::NoFlush | State::Flush),
            "note_log following note_log without note_log_summary in between"
        );
        if matches!(self.state, State::NoFlush) && &self.counter == id {
            self.state = State::IdMatch;
        }
    }

    #[inline]
    fn note_log_summary(&mut self, summary: &ChunkSummary) {
        if matches!(self.state, State::IdMatch) {
            if let ChunkSummary::Counter {
                increment,
                initial_value,
            } = summary
            {
                // The increment is > 0 in most cases anyway so flush on 0 shouldn't be a big risk
                if (increment + initial_value) % self.interval == 0 {
                    self.state = State::Flush;
                } else {
                    self.state = State::NoFlush;
                }
            } else {
                panic!("Target ID {} is not a counter", self.counter);
            }
        }
    }

    #[inline]
    fn flush_group_end(&mut self) -> bool {
        matches!(self.state, State::Flush)
    }

    #[inline]
    fn note_flush(&mut self) {
        self.state = State::NoFlush;
    }
}
