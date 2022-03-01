//! Tensorboard logger
use super::chunk::{ChunkLogger, ChunkSummary, LoggerBackend};
use super::Id;
use std::fmt::{self, Write};
use std::path::Path;
use std::time::Duration;
use tensorboard_rs::summary_writer::SummaryWriter;

/// Logger that saves summaries to a tensorboard file.
pub type TensorBoardLogger = ChunkLogger<TensorBoardBackend>;

impl TensorBoardLogger {
    /// Create a new `TensorBoardLogger`.
    pub fn new<P: AsRef<Path>>(log_dir: P, summary_period: Duration) -> Self {
        Self::from_backend(summary_period, TensorBoardBackend::new(log_dir))
    }
}

/// Logging backend that saves summaries to a tensorboard file.
pub struct TensorBoardBackend {
    writer: SummaryWriter,
    summary_index: usize,
}

impl fmt::Debug for TensorBoardBackend {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("TensorBoardBackend")
            // TODO: Output the log dir if SummaryWriter adds support for reading it.
            // .field("log_dir", &self.writer.get_logdir())
            .finish()
    }
}

impl TensorBoardBackend {
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Self {
        Self {
            writer: SummaryWriter::new(log_dir),
            summary_index: 0,
        }
    }
}

impl LoggerBackend for TensorBoardBackend {
    fn record_summaries<'a, I>(&mut self, summaries: I, elapsed: Duration)
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
            Nothing => {}
            Counter {
                increment,
                initial_value,
            } => {
                self.writer
                    .add_scalar(tag, (initial_value + increment) as f32, self.summary_index)
            }
            Duration { stats } => {
                if let Some(mean) = stats.mean() {
                    self.writer.add_scalar(tag, mean as f32, self.summary_index)
                }
            }
            Scalar { stats } => {
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
            Message { counts: _ } => {}
        }
    }
}
