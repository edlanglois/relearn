//! Statistics utilities
use num_traits::{real::Real, Zero};
use std::fmt;
use std::iter::{Extend, FromIterator};
use std::ops::{Add, AddAssign};

/// Online mean and variance calculation using Welford's Algorithm
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct OnlineMeanVariance<T> {
    mean: T,
    squared_residual_sum: T,
    count: u64,
}

impl<T: Zero> Default for OnlineMeanVariance<T> {
    #[inline]
    fn default() -> Self {
        Self {
            mean: T::zero(),
            squared_residual_sum: T::zero(),
            count: 0,
        }
    }
}

impl<T: Real + fmt::Display> fmt::Display for OnlineMeanVariance<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(μ = ")?;
        if let Some(mean) = self.mean() {
            fmt::Display::fmt(&mean, f)?;
        } else {
            write!(f, "-")?;
        }
        write!(f, "; σ = ")?;
        if let Some(stddev) = self.stddev() {
            fmt::Display::fmt(&stddev, f)?;
        } else {
            write!(f, "-")?;
        }
        write!(f, "; n = ")?;
        fmt::Display::fmt(&self.count, f)?;
        write!(f, ")")
    }
}

impl<T: Zero> OnlineMeanVariance<T> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Copy> OnlineMeanVariance<T> {
    /// The mean of all accumulated values if at least one value has been accumulated.
    #[inline]
    pub fn mean(&self) -> Option<T> {
        if self.count > 0 {
            Some(self.mean)
        } else {
            None
        }
    }

    /// The number of accumulated values.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl<T: Real> OnlineMeanVariance<T> {
    /// The (population) variance of all accumulated values.
    ///
    /// If at least one value has been accumulated.
    #[inline]
    pub fn variance(&self) -> Option<T> {
        if self.count > 0 {
            Some(self.squared_residual_sum / T::from(self.count).unwrap())
        } else {
            None
        }
    }

    /// The (population) standard deviation of all accumulated values.
    ///
    /// If at least one value has been accumulated.
    #[inline]
    pub fn stddev(&self) -> Option<T> {
        self.variance().map(T::sqrt)
    }
}

impl<T: Real> OnlineMeanVariance<T> {
    /// Add a new value to the calculation.
    pub fn push(&mut self, value: T) {
        let residual_pre = value - self.mean;
        self.count += 1;
        self.mean = self.mean + residual_pre / T::from(self.count).unwrap();
        let residual_post = value - self.mean;
        self.squared_residual_sum = self.squared_residual_sum + residual_pre * residual_post;
    }

    /// Calculate mean and variance of a slice using numerically-stable divide-and-conquer
    pub fn from_slice(data: &[T]) -> Self {
        if data.len() <= 8 {
            return data.iter().cloned().collect();
        }
        let mid = data.len() / 2;
        Self::from_slice(&data[..mid]) + Self::from_slice(&data[mid..])
    }

    /// Aggregate mean and variance of slice of `OnlineMeanVariance` instances.
    ///
    /// Uses a numerically-stable divide-and-conquer approach.
    pub fn from_stats(stats: &[Self]) -> Self {
        match stats {
            [] => Self::default(),
            [x] => *x,
            _ => {
                let mid = stats.len();
                Self::from_stats(&stats[..mid]) + Self::from_stats(&stats[mid..])
            }
        }
    }
}

impl<T: Real> Extend<T> for OnlineMeanVariance<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for value in iter {
            self.push(value)
        }
    }
}

impl<T: Real> FromIterator<T> for OnlineMeanVariance<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut s = Self::default();
        s.extend(iter);
        s
    }
}

impl<T: Real> FromIterator<Self> for OnlineMeanVariance<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        iter.into_iter()
            .reduce(|a, b| a + b)
            .unwrap_or_else(Self::default)
    }
}

impl<T: Real> Add for OnlineMeanVariance<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Reference
        // Algorithms for computing the sample variance: analysis and recommendations
        // Chan et al. (1979)
        // <http://www.cs.yale.edu/publications/techreports/tr222.pdf>
        // <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm>
        let self_count = T::from(self.count).unwrap();
        let other_count = T::from(other.count).unwrap();
        let count = self.count + other.count;
        let total_count = T::from(count).unwrap();

        let mean = (self.mean * self_count + other.mean * other_count) / total_count;
        let delta = self.mean - other.mean;
        let squared_residual_sum = self.squared_residual_sum
            + other.squared_residual_sum
            + delta * delta * self_count * other_count / total_count;

        Self {
            mean,
            squared_residual_sum,
            count,
        }
    }
}

impl<T: Real> AddAssign for OnlineMeanVariance<T> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_f64() {
        let stats: OnlineMeanVariance<f64> = [1.0, 2.0, 3.0, 4.0].into_iter().collect();
        assert!((stats.mean().unwrap() - 2.5).abs() < 1e-8);
        assert!((stats.variance().unwrap() - 1.25).abs() < 1e-8);
    }

    #[test]
    fn from_slice() {
        let data: Vec<_> = (0..20u8).map(f32::from).collect();
        let collected: OnlineMeanVariance<_> = data.iter().cloned().collect();
        assert_eq!(OnlineMeanVariance::from_slice(&data), collected);
    }

    #[test]
    fn add() {
        let a: OnlineMeanVariance<f64> = [1.0, 2.0].into_iter().collect();
        let b: OnlineMeanVariance<f64> = [3.0, 4.0].into_iter().collect();
        let c: OnlineMeanVariance<f64> = [1.0, 2.0, 3.0, 4.0].into_iter().collect();
        assert_eq!(a + b, c);
    }

    #[test]
    fn add_assign() {
        let a: OnlineMeanVariance<f64> = [1.0, 2.0].into_iter().collect();
        let mut b: OnlineMeanVariance<f64> = [3.0, 4.0].into_iter().collect();
        b += a;
        let c: OnlineMeanVariance<f64> = [1.0, 2.0, 3.0, 4.0].into_iter().collect();
        assert_eq!(b, c);
    }
}
