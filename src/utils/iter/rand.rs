use rand::Rng;
use std::iter::FusedIterator;

/// Iterator over exactly `num_samples` random sub-samples from a fixed-length iterator.
///
/// Selects values without replacement and in the order they appear in the input iterator.
/// Does not allocate any temporary storage.
/// Is `O(num_samples)` if `Iterator::skip` is `O(1)`.
#[derive(Debug, Clone)]
pub struct RandSubsample<I, R> {
    num_samples: usize,
    iter: I,
    rng: R,
}

impl<I, R> RandSubsample<I, R> {
    pub const fn new(iter: I, num_samples: usize, rng: R) -> Self {
        Self {
            num_samples,
            iter,
            rng,
        }
    }
}

impl<I, R> Iterator for RandSubsample<I, R>
where
    I: ExactSizeIterator,
    R: Rng,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_samples == 0 {
            return None;
        }
        // TODO: Sample num_skip directly.
        // The number of elements to skip is a Negative Hypergeometric Distribution
        // https://en.wikipedia.org/wiki/Negative_hypergeometric_distribution
        // This distribution is not in rand_distr but apparently it is a special case of
        // BetaBinormial and that is in the rv crate.
        let mut remaining = self.iter.len();
        while !self
            .rng
            .gen_bool((self.num_samples as f64 / remaining as f64).min(1.0))
        {
            remaining -= 1;
        }
        let num_skip = self.iter.len() - remaining;

        self.num_samples -= 1;
        self.iter.nth(num_skip)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len().min(self.num_samples);
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<I, R> ExactSizeIterator for RandSubsample<I, R>
where
    I: ExactSizeIterator,
    R: Rng,
{
}

impl<I, R> FusedIterator for RandSubsample<I, R>
where
    I: ExactSizeIterator + FusedIterator,
    R: Rng,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use std::ops::RangeInclusive;

    const POP_SIZE: usize = 50;
    const NUM_SAMPLES: usize = 10;
    /// Number of iterations when checking for a failure
    const NUM_ITERS_CHECK: u64 = 100;
    /// Number of iterations when running a statistical test
    const NUM_ITERS_STATS: u64 = 1000;

    #[test]
    fn rand_subsample_len() {
        let mut rng = StdRng::seed_from_u64(87);
        for _ in 0..NUM_ITERS_CHECK {
            let iter = RandSubsample::new(0..POP_SIZE, NUM_SAMPLES, &mut rng);
            assert_eq!(iter.len(), NUM_SAMPLES);
        }
    }

    #[test]
    fn rand_subsample_count() {
        let mut rng = StdRng::seed_from_u64(87);
        for _ in 0..NUM_ITERS_CHECK {
            let iter = RandSubsample::new(0..POP_SIZE, NUM_SAMPLES, &mut rng);
            assert_eq!(iter.count(), NUM_SAMPLES);
        }
    }

    #[test]
    fn rand_subsample_manual_count() {
        let mut rng = StdRng::seed_from_u64(87);
        for _ in 0..NUM_ITERS_CHECK {
            let iter = RandSubsample::new(0..POP_SIZE, NUM_SAMPLES, &mut rng);
            assert_eq!(iter.map(|_| 1).sum::<usize>(), NUM_SAMPLES);
        }
    }

    #[test]
    fn rand_subsample_unif_dist() {
        let mut rng = StdRng::seed_from_u64(87);
        let mut counts = vec![0; POP_SIZE];
        for _ in 0..NUM_ITERS_STATS {
            let iter = RandSubsample::new(0..POP_SIZE, NUM_SAMPLES, &mut rng);
            for i in iter {
                counts[i] += 1;
            }
        }
        let ci =
            bernoulli_confidence_interval(NUM_SAMPLES as f64 / POP_SIZE as f64, NUM_ITERS_STATS);
        assert!(counts.iter().all(|c| ci.contains(c)));
    }

    #[test]
    fn rand_subsample_sample_all() {
        assert!(RandSubsample::new(0..10, 10, StdRng::seed_from_u64(87)).eq(0..10));
    }

    #[test]
    fn rand_subsample_sample_more() {
        assert!(RandSubsample::new(0..10, 20, StdRng::seed_from_u64(87)).eq(0..10));
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)] // negative f64 casts to 0.0 as desired
    fn bernoulli_confidence_interval(p: f64, n: u64) -> RangeInclusive<u64> {
        // Using Wald method <https://en.wikipedia.org/wiki/Binomial_distribution#Wald_method>
        // Quantile for error rate of 1e-5
        let z = 4.4;
        let nf = n as f64;
        let stddev = (p * (1.0 - p) * nf).sqrt();
        let lower_bound = nf * p - z * stddev;
        let upper_bound = nf * p + z * stddev;
        (lower_bound.round() as u64)..=(upper_bound.round() as u64)
    }
}
