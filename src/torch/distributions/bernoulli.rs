//! Bernoulli distribution
use super::{clamp_float_finite, clamp_float_min};
use crate::utils::distributions::ArrayDistribution;
use once_cell::sync::OnceCell;
use tch::{Reduction, Tensor};

/// Bernoulii distribution(s).
pub struct Bernoulli {
    /// Logits
    ///
    /// An f64 tensor of shape `[BATCH_SHAPE...]`.
    /// `logits_i = log(p_i, / (1 - p_i))`, the logistic function of the probabilities.
    logits: Tensor,
    /// Cached probabilities
    probs: OnceCell<Tensor>,
}

impl Bernoulli {
    /// Initialize from logits
    #[must_use]
    pub fn new(logits: Tensor) -> Self {
        Self {
            logits,
            probs: OnceCell::new(),
        }
    }
}

impl ArrayDistribution<Tensor, Tensor> for Bernoulli {
    fn batch_shape(&self) -> Vec<usize> {
        self.logits
            .size()
            .iter()
            .map(|&s| s.try_into().unwrap()) // convert from i64 to usize
            .collect()
    }

    fn element_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Samples are boolean tensors
    fn sample(&self) -> Tensor {
        self.probs
            .get_or_init(|| self.logits.sigmoid())
            .bernoulli()
            .greater(0.5) // bernoulli() samples are float 0.0 or 1.0
    }

    fn log_probs(&self, elements: &Tensor) -> Tensor {
        -clamp_float_finite(&self.logits)
            .unwrap()
            .binary_cross_entropy_with_logits::<&Tensor>(
                &elements.to_kind(self.logits.kind()),
                None,
                None,
                Reduction::None,
            )
    }

    fn entropy(&self) -> Tensor {
        clamp_float_finite(&self.logits)
            .unwrap()
            .binary_cross_entropy_with_logits::<&Tensor>(
                self.probs.get_or_init(|| self.logits.sigmoid()),
                None,
                None,
                Reduction::None,
            )
    }

    fn kl_divergence_from(&self, other: &Self) -> Tensor {
        let cross_entropy = clamp_float_min(&other.logits)
            .unwrap()
            .binary_cross_entropy_with_logits::<&Tensor>(
                self.probs.get_or_init(|| self.logits.sigmoid()),
                None,
                None,
                Reduction::None,
            );
        cross_entropy - self.entropy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{IndexOp, Kind};

    #[test]
    fn batch_shape_0d() {
        let logits = Tensor::from(2.0);
        let d = Bernoulli::new(logits);
        assert_eq!(d.batch_shape(), [] as [usize; 0]);
    }

    #[test]
    fn batch_shape_1d() {
        let logits = Tensor::of_slice(&[-2.0, 0.0, 1.0]);
        let d = Bernoulli::new(logits);
        assert_eq!(d.batch_shape(), [3]);
    }

    #[test]
    fn element_shape() {
        let logits = Tensor::of_slice(&[-2.0, 0.0, 1.0]);
        let d = Bernoulli::new(logits);
        assert_eq!(d.element_shape(), [] as [usize; 0]);
    }

    #[test]
    fn sample() {
        let logits = Tensor::of_slice(&[f32::NEG_INFINITY, -1000.0, 0.0, 1000.0, f32::INFINITY]);
        let d = Bernoulli::new(logits);
        let samples = d.sample();
        assert_eq!(samples.size(), [5]);
        assert_eq!(samples.kind(), Kind::Bool);

        let samples_vec: Vec<bool> = samples.into();
        assert!(!samples_vec[0]);
        assert!(!samples_vec[1]);
        assert!(samples_vec[3]);
        assert!(samples_vec[4]);
    }

    #[test]
    fn log_probs() {
        // Use f64 in calculations for reduced error
        #[allow(clippy::cast_possible_truncation)]
        fn log_sigmoid(logit: f64) -> f32 {
            -(-logit).exp().ln_1p() as f32
        }

        let logits = Tensor::of_slice(&[f32::NEG_INFINITY, -2.0, 0.0, 1.0, 1.0, f32::INFINITY]);
        let d = Bernoulli::new(logits);
        let log_probs = d.log_probs(&Tensor::of_slice(&[true, true, true, true, false, true]));
        assert_eq!(log_probs.size(), [6]);

        let expected = [
            f32::NEG_INFINITY,
            log_sigmoid(-2.0),
            (0.5f32).ln(),
            log_sigmoid(1.0),
            log_sigmoid(-1.0),
            0.0,
        ];
        // Exclude the first element from comparison, might be NEG_INFINITY or just very negative
        assert!(
            log_probs
                .i(1..)
                .allclose(&Tensor::of_slice(&expected[1..]), 1e-5, 1e-8, false),
            "\nlog_probs: {log_probs:?}\nexpected:  {expected:?}\n"
        );
        // Check that the first element is very negative
        assert!(f32::from(log_probs.i(0)) <= f32::MIN / 2.0);
    }

    #[test]
    fn entropies() {
        // Use f64 in calculations for reduced error
        #[allow(clippy::cast_possible_truncation)]
        fn logit_entropy(logit: f64) -> f32 {
            let p = (1.0 + (-logit).exp()).recip();
            let h = -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
            h as f32
        }

        let logits = Tensor::of_slice(&[f32::NEG_INFINITY, -2.0, 0.0, 1.0, f32::INFINITY]);
        let d = Bernoulli::new(logits);
        let entropies = d.entropy();
        assert_eq!(entropies.size(), [5]);

        let expected = [
            0.0,
            logit_entropy(-2.0),
            -(0.5f32).ln(),
            logit_entropy(1.0),
            0.0,
        ];
        assert!(
            entropies.allclose(&Tensor::of_slice(&expected), 1e-5, 1e-8, false),
            "\nentropies: {entropies:?}\nexpected:  {expected:?}\n"
        );
    }

    #[test]
    fn kl_divergence() {
        // Use f64 in calculations for reduced error
        #[allow(clippy::cast_possible_truncation)]
        fn kl_div(lp: f64, lq: f64) -> f32 {
            // logits to probabilities
            let p = (1.0 + (-lp).exp()).recip();
            let q = (1.0 + (-lq).exp()).recip();
            let kl = p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln();
            kl as f32
        }

        let logit_pairs = [
            (0.0, 0.0),
            (1.0, 2.0),
            (2.0, 1.0),
            (0.0, f32::INFINITY),
            (0.0f32, f32::NEG_INFINITY),
            // Divergence is degenerate with +-INF on LHS
            // (f32::NEG_INFINITY, f32::NEG_INFINITY),
            // (f32::NEG_INFINITY, f32::INFINITY),
            // (f32::NEG_INFINITY, 0.0),
            // (f32::INFINITY, f32::INFINITY),
            // (f32::INFINITY, f32::NEG_INFINITY),
            // (f32::INFINITY, 0.0),
        ];
        let p = Bernoulli::new(Tensor::of_slice(
            &logit_pairs.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
        ));
        let q = Bernoulli::new(Tensor::of_slice(
            &logit_pairs.iter().map(|(_, q)| *q).collect::<Vec<_>>(),
        ));

        let kl_divs = p.kl_divergence_from(&q);
        assert_eq!(kl_divs.size(), [logit_pairs.len() as i64]);

        let expected = [
            0.0,
            kl_div(1.0, 2.0),
            kl_div(2.0, 1.0),
            f32::INFINITY,
            f32::INFINITY,
        ];
        // Check finite values
        assert!(
            kl_divs
                .i(..3)
                .allclose(&Tensor::of_slice(&expected[..3]), 1e-5, 1e-8, false),
            "\nkl_divs: {kl_divs:?}\nexpected: {expected:?}\n"
        );
        // Check that the infinities are very positive
        assert!(f32::from(kl_divs.i(3)) >= f32::MAX / 2.0);
        assert!(f32::from(kl_divs.i(4)) >= f32::MAX / 2.0);
    }
}
