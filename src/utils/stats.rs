//! Statistics utilities
use num_traits::{real::Real, Zero};
use std::iter::{Extend, FromIterator};

/// Online mean and variance calculation using Welford's Algorithm
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct OnlineMeanVariance<T> {
    mean: T,
    squared_residual_sum: T,
    count: u64,
}

impl<T: Zero> Default for OnlineMeanVariance<T> {
    fn default() -> Self {
        Self {
            mean: T::zero(),
            squared_residual_sum: T::zero(),
            count: 0,
        }
    }
}

impl<T: Zero> OnlineMeanVariance<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Copy> OnlineMeanVariance<T> {
    /// The mean of all accumulated values.
    #[inline]
    pub fn mean(&self) -> T {
        self.mean
    }

    /// The number of accumulated samples.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl<T: Real> OnlineMeanVariance<T> {
    /// The (population) variance of all accumulated values.
    pub fn variance(&self) -> T {
        if self.count == 0 {
            T::zero()
        } else {
            self.squared_residual_sum / T::from(self.count).unwrap()
        }
    }

    /// The (population) standard deviation of all accumulated values.
    pub fn stddev(&self) -> T {
        self.variance().sqrt()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_f64() {
        let stats: OnlineMeanVariance<f64> =
            [1.0, 2.0, 3.0, 4.0].into_iter().map(Into::into).collect();
        assert!((stats.mean() - 2.5).abs() < 1e-8);
        assert!((stats.variance() - 1.25).abs() < 1e-8);
    }
}
