//! Formatting utilities
use std::fmt;
use std::time::Duration;

/// Pretty-printing
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// Display a frequency
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Frequency(pub f64);

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
pub struct DisplayFn<F>(pub F)
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
