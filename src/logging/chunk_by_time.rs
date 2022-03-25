use super::chunk::Chunker;
use coarsetime::{Duration as CDuration, Instant as CInstant};
use std::time::Duration;

/// Chunk summaries at fixed time intervals (for [`ChunkLogger`][super::ChunkLogger]).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ByTime {
    // Coarse time is used because the current time is checked on every log event,
    // which might be quite frequent for per-step values so the time checks should be fast.
    // The accuracy of the clock (~1ms for coarse vs. ~1ns for regular) is not very important
    // since it is only used for checking whether the chunk duration has elapsed (~1s).
    pub chunk_duration: CDuration,
    coarse_chunk_start: CInstant,
}

impl ByTime {
    pub fn new(chunk_duration: Duration) -> Self {
        Self {
            chunk_duration: CDuration::new(chunk_duration.as_secs(), chunk_duration.subsec_nanos()),
            coarse_chunk_start: CInstant::now(),
        }
    }
}

impl Default for ByTime {
    fn default() -> Self {
        Self::new(Duration::from_secs(5))
    }
}

impl Chunker for ByTime {
    #[inline]
    fn flush_group_start(&mut self) -> bool {
        // Check whether the chunk duration has elapsed.
        // This is done before logging because logs are likely to occur in bursts with the duration
        // elapsing in between. Logging first would cause the first value of the burst to be split
        // into a separate chunk from the rest.
        self.coarse_chunk_start.elapsed() > self.chunk_duration
    }
    fn note_flush(&mut self) {
        self.coarse_chunk_start = CInstant::now();
    }
}
