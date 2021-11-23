//! Space test utilities
use super::{FeatureSpace, FiniteSpace, SampleSpace};
use crate::utils::num_array::{NumArray1D, NumArray2D};
use ndarray::{ArrayBase, Data, Ix2};
use num_traits::Float;
use rand::prelude::*;
use std::fmt::Debug;

/// Check that space contains samples it generates
pub fn check_contains_samples<S: SampleSpace>(space: &S, num_samples: u32) {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..num_samples {
        let element = space.sample(&mut rng);
        assert!(space.contains(&element));
    }
}

/// Check paired [`FiniteSpace::from_index`] and [`FiniteSpace::to_index`] for each valid index
pub fn check_from_to_index_iter_size<S: FiniteSpace>(space: &S) {
    for index in 0..space.size() {
        let element = space.from_index(index).unwrap();
        assert!(space.contains(&element));
        let index2 = space.to_index(&element);
        assert_eq!(index2, index);
    }
}

/// Check paired [`FiniteSpace::from_index`] and [`FiniteSpace::to_index`] for random valid indices
pub fn check_from_to_index_random<S: FiniteSpace>(space: &S, num_samples: u32) {
    let size = space.size();
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..num_samples {
        let index = rng.gen_range(0..size);
        let element = space.from_index(index).unwrap();
        assert!(space.contains(&element));
        let index2 = space.to_index(&element);
        assert_eq!(index2, index);
    }
}

/// Check [`FiniteSpace::from_index`] for elements sampled randomly from the space.
pub fn check_from_index_sampled<S: FiniteSpace + SampleSpace>(space: &S, num_samples: u32) {
    let mut rng = StdRng::seed_from_u64(3);
    let size = space.size();
    for _ in 0..num_samples {
        let element = space.sample(&mut rng);
        let index = space.to_index(&element);
        assert!(index < size);
    }
}

/// Check [`FiniteSpace::from_index`] for invalid indices.
pub fn check_from_index_invalid<S: FiniteSpace>(space: &S) {
    let size = space.size();
    assert!(space.from_index(size).is_none());
    assert!(space.from_index(size + 1).is_none());
}

/// Check [`FeatureSpace::features`].
pub fn check_features<S, T>(space: &S, element: &S::Element, expected: &[T::Elem])
where
    S: FeatureSpace,
    T: NumArray1D,
    T::Elem: Float + Debug,
{
    let out: T = space.features(element);
    assert_eq!(out.as_slice(), expected);
}

/// Check [`FeatureSpace::features_out`] with a zero-initialized output vector.
pub fn check_features_out_zeroed<S, T>(space: &S, element: &S::Element, expected: &[T::Elem])
where
    S: FeatureSpace,
    T: NumArray1D,
    T::Elem: Float + Debug,
{
    let mut out = T::zeros(expected.len());
    space.features_out(element, out.as_slice_mut(), true);
    assert_eq!(out.as_slice(), expected);
}

/// Check [`FeatureSpace::features_out`] with a non-zero-initialized output vector.
pub fn check_features_out_nonzero<S, T>(space: &S, element: &S::Element, expected: &[T::Elem])
where
    S: FeatureSpace,
    T: NumArray1D,
    T::Elem: Float + Debug,
{
    let mut out = T::ones(expected.len());
    space.features_out(element, out.as_slice_mut(), false);
    assert_eq!(out.as_slice(), expected);
}

/// Check [`FeatureSpace::batch_features`].
pub fn check_batch_features<S, T, D>(
    space: &S,
    elements: &[S::Element],
    expected: &ArrayBase<D, Ix2>,
) where
    S: FeatureSpace,
    T: NumArray2D,
    T::Elem: Float + Debug,
    D: Data<Elem = T::Elem>,
{
    let out: T = space.batch_features(elements);
    assert_eq!(out.view(), expected);
}

/// Check [`FeatureSpace::batch_features_out`] with a zero-initialized output vector.
pub fn check_batch_features_out_zeroed<S, T, D>(
    space: &S,
    elements: &[S::Element],
    expected: &ArrayBase<D, Ix2>,
) where
    S: FeatureSpace,
    T: NumArray2D,
    T::Elem: Float + Debug,
    D: Data<Elem = T::Elem>,
{
    let mut out = T::zeros(expected.dim());
    space.batch_features_out(elements, &mut out.view_mut(), true);
    assert_eq!(out.view(), expected);
}

/// Check [`FeatureSpace::batch_features_out`] with a non-zero-initialized output vector.
pub fn check_batch_features_out_nonzero<S, T, D>(
    space: &S,
    elements: &[S::Element],
    expected: &ArrayBase<D, Ix2>,
) where
    S: FeatureSpace,
    T: NumArray2D,
    T::Elem: Float + Debug,
    D: Data<Elem = T::Elem>,
{
    let mut out = T::ones(expected.dim());
    space.batch_features_out(elements, &mut out.view_mut(), false);
    assert_eq!(out.view(), expected);
}

macro_rules! features_tests {
    ($label:ident, $space:expr, $elem:expr, $expected:expr) => {
        #[allow(unused_imports)]
        mod $label {
            use super::*;
            use crate::spaces::testing;
            use crate::utils::tensor::ExclusiveTensor;
            use ndarray::Array1;

            #[test]
            fn array_features() {
                testing::check_features::<_, Array1<f32>>(&$space, &$elem, &$expected);
            }
            #[test]
            fn array_features_out_zeroed() {
                testing::check_features_out_zeroed::<_, Array1<f32>>(&$space, &$elem, &$expected);
            }
            #[test]
            fn array_features_out_nonzero() {
                testing::check_features_out_nonzero::<_, Array1<f32>>(&$space, &$elem, &$expected);
            }
            #[test]
            fn tensor_features() {
                testing::check_features::<_, ExclusiveTensor<f32, _>>(&$space, &$elem, &$expected);
            }
            #[test]
            fn tensor_features_out_zeroed() {
                testing::check_features_out_zeroed::<_, ExclusiveTensor<f32, _>>(
                    &$space, &$elem, &$expected,
                );
            }
            #[test]
            fn tensor_features_out_nonzero() {
                testing::check_features_out_nonzero::<_, ExclusiveTensor<f32, _>>(
                    &$space, &$elem, &$expected,
                );
            }
        }
    };
}

macro_rules! batch_features_tests {
    ($label:ident, $space:expr, $elems:expr, $expected:expr) => {
        #[allow(unused_imports)]
        mod $label {
            use super::*;
            use crate::spaces::testing;
            use crate::utils::tensor::ExclusiveTensor;
            use ndarray::{arr2, Array2};

            #[test]
            fn array_batch_features() {
                testing::check_batch_features::<_, Array2<f32>, _>(
                    &$space,
                    &$elems,
                    &arr2(&$expected),
                );
            }
            #[test]
            fn array_batch_features_out_zeroed() {
                testing::check_batch_features_out_zeroed::<_, Array2<f32>, _>(
                    &$space,
                    &$elems,
                    &arr2(&$expected),
                );
            }
            #[test]
            fn array_batch_features_out_nonzero() {
                testing::check_batch_features_out_nonzero::<_, Array2<f32>, _>(
                    &$space,
                    &$elems,
                    &arr2(&$expected),
                );
            }
            #[test]
            fn tensor_batch_features() {
                testing::check_batch_features::<_, ExclusiveTensor<f32, _>, _>(
                    &$space,
                    &$elems,
                    &arr2(&$expected),
                );
            }
            #[test]
            fn tensor_batch_features_out_zeroed() {
                testing::check_batch_features_out_zeroed::<_, ExclusiveTensor<f32, _>, _>(
                    &$space,
                    &$elems,
                    &arr2(&$expected),
                );
            }
            #[test]
            fn tensor_batch_features_out_nonzero() {
                testing::check_batch_features_out_nonzero::<_, ExclusiveTensor<f32, _>, _>(
                    &$space,
                    &$elems,
                    &arr2(&$expected),
                );
            }
        }
    };
}
