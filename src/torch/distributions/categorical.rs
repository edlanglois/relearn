//! Categorical distribution
use crate::utils::distributions::ArrayDistribution;
use std::convert::TryInto;
use tch::{Kind, Tensor};

/// Categorical distribution(s).
#[derive(Debug, PartialEq)]
pub struct Categorical {
    /// Log probability of each event.
    ///
    /// An f64 tensor of shape `[BATCH_SHAPE.., NUM_EVENTS]`.
    logits: Tensor,
}

impl Categorical {
    /// Initialze from possibly unnormalized log probabilities.
    ///
    /// The log probabilities are normalized by adding some value `C` to each
    /// such that `sum_i exp(log_prob[i] + C) = 1`.
    pub fn new(logits: &Tensor) -> Self {
        Self {
            logits: logits.log_softmax(-1, Kind::Float),
        }
    }
}

/// Clap float values to be >= the smallest finite float value.
fn clamp_float_min(x: &Tensor) -> Result<Tensor, Kind> {
    match x.kind() {
        Kind::Float => Ok(x.clamp_min(f64::from(f32::MIN))),
        Kind::Double => Ok(x.clamp_min(f64::MIN)),
        kind => Err(kind),
    }
}

impl ArrayDistribution<Tensor, Tensor> for Categorical {
    fn batch_shape(&self) -> Vec<usize> {
        self.logits
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
        self.logits.exp().multinomial(1, true).squeeze_dim(-1)
    }

    fn log_probs(&self, elements: &Tensor) -> Tensor {
        self.logits
            .gather(-1, &elements.unsqueeze(-1), false)
            .squeeze_dim(-1)
    }

    fn entropy(&self) -> Tensor {
        let clamped_logits = clamp_float_min(&self.logits)
            .map_err(|kind| format!("logits must be f32 or f64, not {:?}", kind))
            .unwrap();
        -(clamped_logits * self.logits.exp()).sum_dim_intlist(&[-1], false, Kind::Float)
    }

    fn kl_divergence_from(&self, other: &Self) -> Tensor {
        let clamped_rel_logits = clamp_float_min(&(&self.logits - &other.logits))
            .map_err(|kind| format!("logits must be f32 or f64, not {:?}", kind))
            .unwrap();
        (clamped_rel_logits * self.logits.exp()).sum_dim_intlist(&[-1], false, Kind::Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::IndexOp;

    #[test]
    fn batch_shape_0d() {
        let logits = Tensor::of_slice(&[0.0_f32, 1.0, 0.0]).log();
        let d = Categorical::new(&logits);
        assert_eq!(d.batch_shape(), vec![]);
    }

    #[test]
    fn batch_shape_1d() {
        let logits = Tensor::of_slice(&[
            0.0_f32, 1.0, 0.0, //
            0.2, 0.3, 0.5, //
        ])
        .reshape(&[2, 3])
        .log();
        let d = Categorical::new(&logits);
        assert_eq!(d.batch_shape(), vec![2]);
    }

    #[test]
    fn element_shape() {
        let logits = Tensor::of_slice(&[
            0.0_f32, 1.0, 0.0, //
            0.2, 0.3, 0.5, //
        ])
        .reshape(&[2, 3])
        .log();
        let d = Categorical::new(&logits);
        assert_eq!(d.element_shape(), vec![]);
    }

    #[test]
    fn sample() {
        let logits = Tensor::of_slice(&[
            1.0_f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.3, 0.3, 0.4, //
        ])
        .reshape(&[4, 3])
        .log();
        let d = Categorical::new(&logits);
        let samples = d.sample();
        assert_eq!(samples.size(), vec![4]);
        assert_eq!(samples.i(..3), Tensor::of_slice(&[0_i64, 1, 2]));
        assert!((0..3).contains(&i64::from(samples.i(3))));
    }

    #[test]
    fn log_probs() {
        let logits = Tensor::of_slice(&[
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
        let distribution = Categorical::new(&logits);

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
        let logits = Tensor::of_slice(&[
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
        let distribution = Categorical::new(&logits);

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
        let logits_a = Tensor::of_slice(&[
            0.2_f32, 0.3, 0.5, //
            0.2, 0.3, 0.5, //
            0.0, 1.0, 0.0, //
        ])
        .reshape(&[3, 3])
        .log();
        let distribution_a = Categorical::new(&logits_a);

        let logits_b = Tensor::of_slice(&[
            0.2_f32, 0.3, 0.5, //
            0.7, 0.2, 0.1, //
            0.2, 0.3, 0.5, //
        ])
        .reshape(&[3, 3])
        .log();
        let distribution_b = Categorical::new(&logits_b);

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
