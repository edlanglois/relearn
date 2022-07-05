//! Command-line logger
use super::chunk::{ChunkLogger, ChunkSummary, Chunker, SummaryWriter};
use super::{ByTime, Id, LogError, LogValue, StatsLogger};
use crate::utils::fmt::{DisplayFn, Frequency, PrettyPrint};
use std::fmt;
use std::time::Duration;
use yansi::Paint;

/// Logger that displays grouped summaries to standard output.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct DisplayLogger<C: Chunker = ByTime>(ChunkLogger<C, DisplayBackend>);

impl<C: Chunker> DisplayLogger<C> {
    #[inline]
    pub fn new(chunker: C) -> Self {
        Self(ChunkLogger::new(chunker, DisplayBackend))
    }
}

impl<C: Chunker> StatsLogger for DisplayLogger<C> {
    #[inline]
    fn group_start(&mut self) {
        self.0.group_start()
    }
    #[inline]
    fn group_log(&mut self, id: Id, value: LogValue) -> Result<(), LogError> {
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

/// Logging backend that displays summaries to standard output.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DisplayBackend;

impl SummaryWriter for DisplayBackend {
    fn write_summaries<'a, I>(&mut self, summaries: I, elapsed: Duration)
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
                    if first {
                        first = false;
                    } else {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", c * 100 / n)?;
                }
                write!(f, "]%")
            }
        }
    }
}

/// Divide a `Duration` by `u64`
fn duration_div_u64(d: Duration, x: u64) -> Duration {
    // Cannot directly translate div_u32 to div_u64 because there might be overflow.
    // Instead use float division if the divisor cannot be converted to u32.
    if let Ok(x32) = u32::try_from(x) {
        d / x32
    } else {
        d.div_f64(x as f64)
    }
}
