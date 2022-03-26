//! Categorical distribution
use super::clamp_float_min;
use crate::utils::distributions::ArrayDistribution;
use tch::{Kind, Tensor};

/// Categorical distribution(s).
#[derive(Debug, PartialEq)]
pub struct Categorical {
    /// Normalized log probability of each outcome.
    ///
    /// An f64 tensor of shape `[BATCH_SHAPE.., NUM_EVENTS]`.
    ///
    /// Note: PyTorch refers to these as `logits` but I am not sure that is correct,
    /// or at least it is confusing given the conventional meaning of Bernoulli logits.
    /// These are `log(p_i)` but Bernoulli logits are `log(p_i / (1 - p_i))`.
    /// The multinomial equivalent of logit parameterization omits one of the redundant log probs
    /// (say 0) and uses `log(p_i / p_0)`.
    /// For example, see
    /// <https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_set_of_independent_binary_regressions>
    log_probs: Tensor,
}

impl Categorical {
    /// Initialze from unnormalized log probabilities.
    ///
    /// The unnormalized input log probabilities `log(q_i)` are normalized as
    /// `log(p_i) = log(q_i / sum_i(q_i))`.
    #[must_use]
    pub fn new(unnormalized_log_probs: &Tensor) -> Self {
        Self {
            log_probs: unnormalized_log_probs.log_softmax(-1, Kind::Float),
        }
    }
}

impl ArrayDistribution<Tensor, Tensor> for Categorical {
    fn batch_shape(&self) -> Vec<usize> {
        self.log_probs
            .size() // shape as i64
            .split_last() // exclude NUM_EVENTS dim
            .unwrap()
            .1
            .iter()
            .map(|&s| s.try_into().unwrap()) // convert to usize
            .collect()
    }

    fn element_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    fn sample(&self) -> Tensor {
        self.log_probs.exp().multinomial(1, true).squeeze_dim(-1)
    }

    fn log_probs(&self, elements: &Tensor) -> Tensor {
        self.log_probs
            .gather(-1, &elements.unsqueeze(-1), false)
            .squeeze_dim(-1)
    }

    fn entropy(&self) -> Tensor {
        // Clamping avoids -INF * exp(-INF) = -INF * 0 = NaN
        let clamped_log_probs = clamp_float_min(&self.log_probs)
            .map_err(|kind| format!("log_probs must be f32 or f64, not {:?}", kind))
            .unwrap();
        -(clamped_log_probs * self.log_probs.exp()).sum_dim_intlist(&[-1], false, Kind::Float)
    }

    fn kl_divergence_from(&self, other: &Self) -> Tensor {
        // Clamping avoids -INF * exp(-INF) = -INF * 0 = NaN
        let clamped_rel_log_probs = clamp_float_min(&(&self.log_probs - &other.log_probs))
            .map_err(|kind| format!("log_probs must be f32 or f64, not {:?}", kind))
            .unwrap();
        (clamped_rel_log_probs * self.log_probs.exp()).sum_dim_intlist(&[-1], false, Kind::Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::IndexOp;

    #[test]
    fn batch_shape_0d() {
        let log_probs = Tensor::of_slice(&[0.0_f32, 1.0, 0.0]).log();
        let d = Categorical::new(&log_probs);
        assert_eq!(d.batch_shape(), [] as [usize; 0]);
    }

    #[test]
    fn batch_shape_1d() {
        let log_probs = Tensor::of_slice(&[
            0.0_f32, 1.0, 0.0, //
            0.2, 0.3, 0.5, //
        ])
        .reshape(&[2, 3])
        .log();
        let d = Categorical::new(&log_probs);
        assert_eq!(d.batch_shape(), [2]);
    }

    #[test]
    fn element_shape() {
        let log_probs = Tensor::of_slice(&[
            0.0_f32, 1.0, 0.0, //
            0.2, 0.3, 0.5, //
        ])
        .reshape(&[2, 3])
        .log();
        let d = Categorical::new(&log_probs);
        assert_eq!(d.element_shape(), [] as [usize; 0]);
    }

    #[test]
    fn sample() {
        let log_probs = Tensor::of_slice(&[
            1.0_f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.3, 0.3, 0.4, //
        ])
        .reshape(&[4, 3])
        .log();
        let d = Categorical::new(&log_probs);
        let samples = d.sample();
        assert_eq!(samples.size(), [4]);
        assert_eq!(samples.i(..3), Tensor::of_slice(&[0_i64, 1, 2]));
        assert!((0..3).contains(&i64::from(samples.i(3))));
    }

    #[test]
    fn log_probs() {
        let log_probs = Tensor::of_slice(&[
            // elem: 1
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            // elem: 0
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            // elem: 2
            f32::NEG_INFINITY,
            0.0,
            0.0,
            // elem: 0
            f32::NEG_INFINITY,
            0.0,
            0.0,
            // elem: 0
            -1.0,
            0.0,
            1.0,
            // elem: 1
            -1.0,
            0.0,
            1.0,
            // elem: 2
            -1.0,
            0.0,
            1.0,
            // elem: 0
            0.0,
            0.0,
            0.0,
        ])
        .reshape(&[-1, 3]);
        let distribution = Categorical::new(&log_probs);

        let elements = Tensor::of_slice(&[1_i64, 0, 2, 0, 0, 1, 2, 0]);

        // Log normalizing constant for the [-1, 0.0, 1] distribution
        let log_normalizer = f32::ln(f32::exp(-1.0) + 1.0 + f32::exp(1.0));
        let expected = Tensor::of_slice(&[
            0.0,
            f32::NEG_INFINITY,
            -f32::ln(2.0),
            f32::NEG_INFINITY,
            -1.0 - log_normalizer,
            -log_normalizer,
            1.0 - log_normalizer,
            f32::ln(3.0_f32.recip()),
        ]);

        let actual = distribution.log_probs(&elements);

        assert!(
            Into::<bool>::into(expected.isclose(&actual, 1e-6, 1e-6, false).all()),
            "expected: {:?}\nactual: {:?}",
            expected,
            actual
        );
    }

    #[test]
    fn entropies() {
        let log_probs = Tensor::of_slice(&[
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            //
            f32::NEG_INFINITY,
            0.0,
            0.0,
            //
            0.0,
            0.0,
            0.0,
            //
            0.1_f32.ln(),
            0.3_f32.ln(),
            0.6_f32.ln(),
        ])
        .reshape(&[-1, 3]);
        let distribution = Categorical::new(&log_probs);

        let actual = distribution.entropy();
        let expected = Tensor::of_slice(&[
            0.0,
            -(0.5_f32.ln()),
            -(3.0_f32.recip().ln()),
            -0.1 * 0.1_f32.ln() - 0.3 * 0.3_f32.ln() - 0.6 * 0.6_f32.ln(),
        ]);

        assert!(
            Into::<bool>::into(expected.isclose(&actual, 1e-6, 1e-6, false).all()),
            "expected: {:?}\nactual: {:?}",
            expected,
            actual
        );
    }

    #[test]
    fn kl_divergence() {
        let log_probs_a = Tensor::of_slice(&[
            0.2_f32, 0.3, 0.5, //
            0.2, 0.3, 0.5, //
            0.0, 1.0, 0.0, //
        ])
        .reshape(&[3, 3])
        .log();
        let distribution_a = Categorical::new(&log_probs_a);

        let log_probs_b = Tensor::of_slice(&[
            0.2_f32, 0.3, 0.5, //
            0.7, 0.2, 0.1, //
            0.2, 0.3, 0.5, //
        ])
        .reshape(&[3, 3])
        .log();
        let distribution_b = Categorical::new(&log_probs_b);

        let actual = distribution_a.kl_divergence_from(&distribution_b);
        let expected = Tensor::of_slice(&[
            0.0_f32,
            0.2 * (0.2_f32 / 0.7).ln() + 0.3 * (0.3_f32 / 0.2).ln() + 0.5 * (0.5_f32 / 0.1).ln(),
            (1.0_f32 / 0.3).ln(),
        ]);

        assert!(
            Into::<bool>::into(expected.isclose(&actual, 1e-6, 1e-6, false).all()),
            "expected: {:?}\nactual: {:?}",
            expected,
            actual
        );
    }
}
