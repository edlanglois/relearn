//! Tensorboard logger
use super::chunk::{ChunkLogger, ChunkSummary, Chunker, SummaryWriter};
use super::{ByTime, Id, LogError, Loggable, StatsLogger};
use std::fmt::{self, Write};
use std::path::Path;
use std::time::Duration;
use tensorboard_rs::summary_writer::SummaryWriter as TbSummaryWriter;

/// Logger that saves grouped summaries to a tensorboard file.
#[derive(Debug)]
pub struct TensorBoardLogger<C: Chunker = ByTime>(ChunkLogger<C, TensorBoardBackend>);

impl<C: Chunker> TensorBoardLogger<C> {
    #[inline]
    pub fn new<P: AsRef<Path>>(chunker: C, log_dir: P) -> Self {
        Self(ChunkLogger::new(chunker, TensorBoardBackend::new(log_dir)))
    }
}

impl<C: Chunker> StatsLogger for TensorBoardLogger<C> {
    #[inline]
    fn group_start(&mut self) {
        self.0.group_start()
    }
    #[inline]
    fn group_log(&mut self, id: Id, value: Loggable) -> Result<(), LogError> {
        self.0.group_log(id, value)
    }
    #[inline]
    fn group_end(&mut self) {
        self.0.group_end()
    }
    #[inline]
    fn flush(&mut self) {
        self.0.flush()
    }
}

/// Logging backend that saves summaries to a tensorboard file.
pub struct TensorBoardBackend {
    writer: TbSummaryWriter,
    summary_index: usize,
}

impl fmt::Debug for TensorBoardBackend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("TensorBoardBackend")
            // TODO: Output the log dir if TbSummaryWriter adds support for reading it.
            // .field("log_dir", &self.writer.get_logdir())
            .finish()
    }
}

impl TensorBoardBackend {
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Self {
        Self {
            writer: TbSummaryWriter::new(log_dir),
            summary_index: 0,
        }
    }
}

impl SummaryWriter for TensorBoardBackend {
    fn write_summaries<'a, I>(&mut self, summaries: I, elapsed: Duration)
    where
        I: Iterator<Item = (&'a Id, &'a ChunkSummary)>,
    {
        // NOTE:
        // The writer methods copy the given &str into a String.
        // It would be better if they took a string instead so a double-copy wouldn't be necessary.
        // Using a persistent buffer on our end at least prevents double allocations.
        let mut tag_buffer = String::new();

        for (id, summary) in summaries {
            tag_buffer.clear();
            write!(tag_buffer, "{}", id).unwrap();
            self.write_summary(&tag_buffer, summary, elapsed);
        }
        self.summary_index += 1;
        self.writer.flush();
    }
}

impl TensorBoardBackend {
    fn write_summary(&mut self, tag: &str, summary: &ChunkSummary, _: Duration) {
        use ChunkSummary::*;

        #[allow(clippy::cast_possible_truncation)]
        match summary {
            Counter {
                increment,
                initial_value,
            } => {
                self.writer
                    .add_scalar(tag, (initial_value + increment) as f32, self.summary_index)
            }
            Duration { stats } | Scalar { stats } => {
                if let Some(mean) = stats.mean() {
                    self.writer.add_scalar(tag, mean as f32, self.summary_index)
                }
            }
            Index { counts } => {
                // Treat as a histogram with bucket boundaries half way between each integer.
                self.writer.add_histogram_raw(
                    tag,
                    -0.5,                                                         // min
                    counts.len() as f64 - 0.5,                                    // max
                    counts.iter().map(|n| *n as f64).sum(),                       // num
                    counts.iter().enumerate().map(|(i, n)| (i * n) as f64).sum(), // sum
                    counts
                        .iter()
                        .enumerate()
                        .map(|(i, n)| (i * i * n) as f64)
                        .sum(), // sum_squares
                    &(0..counts.len())
                        .map(|i| i as f64 + 0.5)
                        .collect::<Vec<_>>(), // bucket_limits
                    &counts.iter().map(|n| *n as f64).collect::<Vec<_>>(),        // bucket counts
                    self.summary_index,
                )
            }
            Nothing | Message { counts: _ } => {}
        }
    }
}
