//! FiniteSpace trait definition
use super::{FeatureSpace, ParameterizedSampleSpace, ReprSpace, Space};
use crate::torch::utils as torch_utils;
use tch::{Device, Kind, Tensor};

/// A space containing finitely many elements.
pub trait FiniteSpace: Space {
    /// The number of elements in the space.
    fn size(&self) -> usize;

    /// Get the index of an element.
    fn to_index(&self, element: &Self::Element) -> usize;

    /// Try to convert an index to an element.
    ///
    /// If None is returned then the index was invalid.
    /// It is allowed that Some value may be returned even if the index is invalid.
    /// If you need to validate the returned value, use contains().
    fn from_index(&self, index: usize) -> Option<Self::Element>;
}

/// Represents elements as integer tensors.
impl<S: FiniteSpace> ReprSpace<Tensor> for S {
    fn repr(&self, element: &Self::Element) -> Tensor {
        Tensor::scalar_tensor(self.to_index(element) as i64, (Kind::Int64, Device::Cpu))
    }

    fn batch_repr<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let indices: Vec<_> = elements
            .into_iter()
            .map(|elem| self.to_index(elem) as i64)
            .collect();
        Tensor::of_slice(&indices)
    }
}

/// Represents elements with one-hot feature vectors.
impl<S: FiniteSpace> FeatureSpace<Tensor> for S {
    fn num_features(&self) -> usize {
        self.size()
    }

    fn features(&self, element: &Self::Element) -> Tensor {
        torch_utils::one_hot(
            &Tensor::scalar_tensor(self.to_index(&element) as i64, (Kind::Int64, Device::Cpu)),
            self.num_features(),
            Kind::Float,
        )
    }

    fn batch_features<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let indices: Vec<_> = elements
            .into_iter()
            .map(|element| self.to_index(element) as i64)
            .collect();
        torch_utils::one_hot(
            &Tensor::of_slice(&indices),
            self.num_features(),
            Kind::Float,
        )
    }
}

/// Parameterize a categorical distribution.
impl<S: FiniteSpace> ParameterizedSampleSpace<Tensor> for S {
    fn num_sample_params(&self) -> usize {
        self.size()
    }

    fn sample(&self, parameters: &Tensor) -> Self::Element {
        self.from_index(
            Into::<i64>::into(parameters.softmax(-1, Kind::Float).multinomial(1, true)) as usize,
        )
        .unwrap()
    }

    fn batch_log_probs(&self, parameters: &Tensor, elements: &Tensor) -> Tensor {
        let logits = parameters.log_softmax(-1, Kind::Float);
        logits
            .gather(-1, &elements.unsqueeze(-1), false)
            .squeeze1(-1)
    }

    fn batch_statistics(&self, parameters: &Tensor, elements: &Tensor) -> (Tensor, Tensor) {
        let logits = parameters.log_softmax(-1, Kind::Float);
        let log_probs = logits
            .gather(-1, &elements.unsqueeze(-1), false)
            .squeeze1(-1);
        let entropy = -(&logits * logits.exp()).sum1(&[-1], false, Kind::Float);
        (log_probs, entropy)
    }
}

#[cfg(test)]
mod trit {
    use rust_rl_derive::Indexed;

    #[derive(Debug, Indexed, PartialEq, Eq)]
    pub enum Trit {
        Zero,
        One,
        Two,
    }
}

#[cfg(test)]
mod repr_space_tensor {
    use super::super::IndexedTypeSpace;
    use super::trit::Trit;
    use super::*;

    #[test]
    fn repr() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        assert_eq!(
            space.repr(&Trit::Zero),
            Tensor::scalar_tensor(0, (Kind::Int64, Device::Cpu))
        );
        assert_eq!(
            space.repr(&Trit::One),
            Tensor::scalar_tensor(1, (Kind::Int64, Device::Cpu))
        );
        assert_eq!(
            space.repr(&Trit::Two),
            Tensor::scalar_tensor(2, (Kind::Int64, Device::Cpu))
        );
    }

    #[test]
    fn batch_repr() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let elements = [Trit::Zero, Trit::One, Trit::Two, Trit::One];
        let actual = space.batch_repr(&elements);
        let expected = Tensor::of_slice(&[0i64, 1, 2, 1]);
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod feature_space_tensor {
    use super::super::IndexedTypeSpace;
    use super::trit::Trit;
    use super::*;

    #[test]
    fn num_features() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        assert_eq!(3, space.num_features());
    }

    #[test]
    fn features() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        assert_eq!(
            Tensor::of_slice(&[1.0, 0.0, 0.0]),
            space.features(&Trit::Zero)
        );
        assert_eq!(
            Tensor::of_slice(&[0.0, 1.0, 0.0]),
            space.features(&Trit::One)
        );
        assert_eq!(
            Tensor::of_slice(&[0.0, 0.0, 1.0]),
            space.features(&Trit::Two)
        );
    }

    #[test]
    fn batch_features() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        assert_eq!(
            Tensor::of_slice(&[
                0.0, 0.0, 1.0, // Two
                1.0, 0.0, 0.0, // Zero
                0.0, 1.0, 0.0, // One
                1.0, 0.0, 0.0 // Zero
            ])
            .reshape(&[4, 3]),
            space.batch_features(&[Trit::Two, Trit::Zero, Trit::One, Trit::Zero])
        );
    }
}

#[cfg(test)]
mod parameterized_sample_space_tensor {
    use super::super::IndexedTypeSpace;
    use super::trit::Trit;
    use super::*;

    #[test]
    fn num_sample_params() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        assert_eq!(3, space.num_sample_params());
    }

    #[test]
    fn sample_deterministic() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let params = Tensor::of_slice(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]);
        for _ in 0..10 {
            assert_eq!(Trit::One, space.sample(&params));
        }
    }

    #[test]
    fn sample_two_of_three() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let params = Tensor::of_slice(&[f32::NEG_INFINITY, 0.0, 0.0]);
        for _ in 0..10 {
            assert!(Trit::Zero != space.sample(&params));
        }
    }

    #[test]
    fn sample_check_distribution() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        // Probabilities: [0.09, 0.24, 0.67]
        let params = Tensor::of_slice(&[-1.0, 0.0, 1.0]);
        let mut one_count = 0;
        let mut two_count = 0;
        let mut three_count = 0;
        for _ in 0..1000 {
            match space.sample(&params) {
                Trit::Zero => one_count += 1,
                Trit::One => two_count += 1,
                Trit::Two => three_count += 1,
            }
        }
        // Check that the counts are within 3.5 standard deviations of the mean
        assert!((58..=121).contains(&one_count));
        assert!((197..=292).contains(&two_count));
        assert!((613..=717).contains(&three_count));
    }

    #[test]
    fn batch_log_probs() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let params = Tensor::of_slice(&[
            // elem: One
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            // elem: Zero
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            // elem: Two
            f32::NEG_INFINITY,
            0.0,
            0.0,
            // elem: Zero
            f32::NEG_INFINITY,
            0.0,
            0.0,
            // elem: Zero
            -1.0,
            0.0,
            1.0,
            // elem: One
            -1.0,
            0.0,
            1.0,
            // elem: Two
            -1.0,
            0.0,
            1.0,
        ])
        .reshape(&[-1, 3]);
        let elements_list = [
            Trit::One,
            Trit::Zero,
            Trit::Two,
            Trit::Zero,
            Trit::Zero,
            Trit::One,
            Trit::Two,
        ];
        let elements = space.batch_repr(&elements_list);

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
        ]);

        let actual = space.batch_log_probs(&params, &elements);

        assert!(Into::<bool>::into(
            expected.isclose(&actual, 1e-6, 1e-6, false).all()
        ));
    }
}
