//! Command-line logger
use super::chunk::{ChunkLogger, ChunkSummary, LoggerBackend, DEFAULT_CHUNK_SECONDS};
use super::Id;
use crate::utils::fmt::{DisplayFn, Frequency, PrettyPrint};
use std::fmt;
use std::time::Duration;
use yansi::Paint;

/// Configuration for [`DisplayLogger`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DisplayLoggerConfig {
    /// Display and summary period.
    pub display_period: Duration,
}

impl Default for DisplayLoggerConfig {
    fn default() -> Self {
        Self {
            display_period: Duration::from_secs(DEFAULT_CHUNK_SECONDS),
        }
    }
}

impl DisplayLoggerConfig {
    pub fn build_logger(&self) -> DisplayLogger {
        DisplayLogger::new(self.display_period)
    }
}

/// Logger that displays summaries to standard output.
pub type DisplayLogger = ChunkLogger<DisplayBackend>;

impl DisplayLogger {
    /// Create a new `DisplayLogger`
    ///
    /// # Args
    /// * `display_period` - Display and summary period.
    pub fn new(display_period: Duration) -> Self {
        Self::from_backend(display_period, DisplayBackend)
    }
}

/// Logging backend that displays summaries to standard output.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DisplayBackend;

impl LoggerBackend for DisplayBackend {
    fn record_summaries<'a, I>(&mut self, summaries: I, elapsed: Duration)
    where
        I: Iterator<Item = (&'a Id, &'a ChunkSummary)>,
    {
        let elapsed = &elapsed;
        println!();
        for (id, summary) in summaries {
            println!(
                "{:<24} {}",
                Paint::fixed(35, id),
                DisplaySummary { summary, elapsed }
            );
        }
    }
}

#[derive(Debug)]
struct DisplaySummary<'a> {
    summary: &'a ChunkSummary,
    elapsed: &'a Duration,
}

impl<'a> fmt::Display for DisplaySummary<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.summary {
            ChunkSummary::Nothing => Ok(()),
            ChunkSummary::Counter {
                increment,
                initial_value,
            } => {
                write!(
                    f,
                    "{}  (+{})",
                    initial_value + increment,
                    Paint::fixed(253, increment)
                )?;
                if *increment > 5 {
                    // Not very accurate unless have several increments in this chunk
                    // because part of the time might have been outside of this chunk
                    let period = duration_div_u64(*self.elapsed, *increment);
                    write!(
                        f,
                        "  {:.2}  {:.3}",
                        Paint::fixed(111, Frequency::from_period(period)),
                        Paint::fixed(111, PrettyPrint(period))
                    )?;
                }
                Ok(())
            }
            ChunkSummary::Duration { stats } => {
                if stats.count() > 0 {
                    let mean = stats.mean().unwrap();
                    write!(f, "{:.4}", PrettyPrint(Duration::from_secs_f64(mean)))?;
                    if stats.count() > 1 {
                        write!(
                            f,
                            " {}",
                            Paint::fixed(
                                8,
                                DisplayFn(|f| write!(
                                    f,
                                    "(σ {:.4})",
                                    PrettyPrint(Duration::from_secs_f64(stats.stddev().unwrap()))
                                ))
                            )
                        )?;
                    }
                    write!(
                        f,
                        " {}",
                        Paint::fixed(
                            221,
                            DisplayFn(|f| write!(
                                f,
                                "{:.2}%",
                                mean / self.elapsed.as_secs_f64() * 100.0
                            ))
                        )
                    )?;
                }
                Ok(())
            }
            ChunkSummary::Scalar { stats } => {
                if stats.count() > 0 {
                    write!(f, "{:.3}", PrettyPrint(stats.mean().unwrap()))?;
                    if stats.count() > 1 {
                        write!(
                            f,
                            " {}",
                            Paint::fixed(
                                8,
                                DisplayFn(|f| write!(
                                    f,
                                    "(σ {:.3})",
                                    PrettyPrint(stats.stddev().unwrap())
                                ))
                            )
                        )?;
                    }
                }
                Ok(())
            }
            ChunkSummary::Index { counts } => {
                let n: usize = counts.iter().sum();
                write!(f, "(n {})  [", n)?;
                let mut first = true;
                for c in counts {
                    if !first {
                        write!(f, " ")?;
                    } else {
                        first = false;
                    }
                    write!(f, "{}", c * 100 / n)?;
                }
                write!(f, "]%")
            }
            ChunkSummary::Message { counts } => {
                for (msg, count) in counts.iter() {
                    write!(f, "\n\t{}  \"{}\"", count, msg)?;
                }
                Ok(())
            }
        }
    }
}

/// Divide a `Duration` by `u64`
fn duration_div_u64(d: Duration, x: u64) -> Duration {
    // Cannot directly translate div_u32 to div_u64 because there might be overflow.
    // Instead use float division if the divisor cannot be converted to u32.
    if let Ok(x32) = x.try_into() {
        d / x32
    } else {
        d.div_f64(x as f64)
    }
}
