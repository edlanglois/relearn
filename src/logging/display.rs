//! Command-line logger
use super::chunk::{ChunkLogger, ChunkSummary, LoggerBackend};
use super::Id;
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
            display_period: Duration::from_secs(1),
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
                let mean = stats.mean();
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
                                PrettyPrint(Duration::from_secs_f64(stats.stddev()))
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
                )
            }
            ChunkSummary::Scalar { stats } => {
                write!(f, "{:.3}", PrettyPrint(stats.mean()))?;
                if stats.count() > 1 {
                    write!(
                        f,
                        " {}",
                        Paint::fixed(
                            8,
                            DisplayFn(|f| write!(f, "(σ {:.3})", PrettyPrint(stats.stddev())))
                        )
                    )?;
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

/// Pretty-printing
#[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd)]
pub struct PrettyPrint<T>(pub T);

impl fmt::Display for PrettyPrint<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let magnitude = self.0.abs();
        if (magnitude >= 1e6 || magnitude <= 1e-4) && self.0 != 0.0 {
            fmt::LowerExp::fmt(&self.0, f)
        } else {
            fmt::Display::fmt(&self.0, f)
        }
    }
}

impl fmt::Display for PrettyPrint<Duration> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The built-in debug output works
        fmt::Debug::fmt(&self.0, f)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Frequency(f64);

impl Frequency {
    pub fn from_period(period: Duration) -> Self {
        Self(period.as_secs_f64().recip())
    }
}

impl fmt::Display for Frequency {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let value = self.0;
        // Half-open ranges in match statements are not yet stable; have to use if instead.
        let (coef, unit) = if (1e3..1e6).contains(&value) {
            (value / 1e3, "kHz")
        } else if (1e6..1e9).contains(&value) {
            (value / 1e6, "MHz")
        } else if (1e9..1e12).contains(&value) {
            (value / 1e9, "GHz")
        } else {
            (value, "Hz")
        };
        fmt::Display::fmt(&PrettyPrint(coef), f)?;
        f.write_str(unit)
    }
}

/// Wraps a closure as the Display implementation
#[derive(Debug)]
pub struct DisplayFn<F>(F)
where
    // Bounded here so that the closure type does not have to be specified on creation
    F: Fn(&mut fmt::Formatter) -> fmt::Result;

impl<F> fmt::Display for DisplayFn<F>
where
    F: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (self.0)(f)
    }
}

const NANOS_PER_SEC: u64 = 1_000_000_000;

/// Divide a `Duration` by `u64`
fn duration_div_u64(d: Duration, x: u64) -> Duration {
    assert_ne!(x, 0);
    let d_secs = d.as_secs();
    let d_nanos = d.subsec_nanos();

    let secs = d_secs / x;
    let carry = d_secs - secs * x;
    let extra_nanos = carry * NANOS_PER_SEC / x;
    let nanos_u64 = u64::from(d_nanos) / x + extra_nanos;
    debug_assert!(nanos_u64 < NANOS_PER_SEC);
    // NANOS_PER_SEC is less than u32::SIZE so must fit
    let nanos: u32 = nanos_u64.try_into().unwrap();
    Duration::new(secs, nanos)
}
