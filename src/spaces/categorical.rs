//! Categorical subtype of [`FiniteSpace`]
use super::{ElementRefInto, FeatureSpace, FiniteSpace, ParameterizedDistributionSpace, ReprSpace};
use crate::logging::Loggable;
use crate::torch;
use crate::utils::distributions::ArrayDistribution;
use std::convert::TryInto;
use tch::{Device, Kind, Tensor};

/// A space consisting of `N` distinct elements treated as distinct and unrelated.
///
/// This does not assume any particular internal structure for the space.
///
/// Implementing this trait provides implementations for:
/// * [`ReprSpace<Tensor>`] as an integer index
/// * [`FeatureSpace<Tensor>`] as a one-hot vector
/// * [`ParameterizedDistributionSpace<Tensor>`] as a categorical distribution
/// * [`ElementRefInto<Loggable>`] as [`Loggable::IndexSample`]
pub trait CategoricalSpace: FiniteSpace {}

/// Represents elements as integer tensors.
impl<S: CategoricalSpace> ReprSpace<Tensor> for S {
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
impl<S: CategoricalSpace> FeatureSpace<Tensor> for S {
    fn num_features(&self) -> usize {
        self.size()
    }

    fn features(&self, element: &Self::Element) -> Tensor {
        torch::utils::one_hot(
            &Tensor::scalar_tensor(self.to_index(element) as i64, (Kind::Int64, Device::Cpu)),
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
        torch::utils::one_hot(
            &Tensor::of_slice(&indices),
            self.num_features(),
            Kind::Float,
        )
    }
}

/// Parameterize a categorical distribution.
impl<S: CategoricalSpace> ParameterizedDistributionSpace<Tensor> for S {
    type Distribution = torch::distributions::Categorical;

    fn num_distribution_params(&self) -> usize {
        self.size()
    }

    fn sample_element(&self, params: &Tensor) -> Self::Element {
        self.from_index(
            self.distribution(params)
                .sample()
                .int64_value(&[])
                .try_into()
                .unwrap(),
        )
        .unwrap()
    }

    fn distribution(&self, params: &Tensor) -> Self::Distribution {
        Self::Distribution::new(params)
    }
}

/// Log the index as a sample from `0..N`
impl<S: CategoricalSpace> ElementRefInto<Loggable> for S {
    fn elem_ref_into(&self, element: &Self::Element) -> Loggable {
        Loggable::IndexSample {
            value: self.to_index(element),
            size: self.size(),
        }
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
        let expected = Tensor::of_slice(&[0_i64, 1, 2, 1]);
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
        assert_eq!(3, space.num_distribution_params());
    }

    #[test]
    fn sample_element_deterministic() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let params = Tensor::of_slice(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]);
        for _ in 0..10 {
            assert_eq!(Trit::One, space.sample_element(&params));
        }
    }

    #[test]
    fn sample_element_two_of_three() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let params = Tensor::of_slice(&[f32::NEG_INFINITY, 0.0, 0.0]);
        for _ in 0..10 {
            assert!(Trit::Zero != space.sample_element(&params));
        }
    }

    #[test]
    fn sample_element_check_distribution() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        // Probabilities: [0.09, 0.24, 0.67]
        let params = Tensor::of_slice(&[-1.0, 0.0, 1.0]);
        let mut one_count = 0;
        let mut two_count = 0;
        let mut three_count = 0;
        for _ in 0..1000 {
            match space.sample_element(&params) {
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
}