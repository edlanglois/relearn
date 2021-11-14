//! Option space definition.
use super::{ElementRefInto, EncoderFeatureSpace, FiniteSpace, NumFeatures, Space};
use crate::logging::Loggable;
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use std::fmt;

/// A space whose elements are either `None` or `Some(inner_elem)`.
///
/// The feature vectors are
/// * `[1, 0, ..., 0]` for `None`
/// * `[0, inner_feature_vector(x)]` for `Some(x)`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OptionSpace<S> {
    pub inner: S,
}

impl<S> OptionSpace<S> {
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: fmt::Display> fmt::Display for OptionSpace<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OptionSpace<{}>", self.inner)
    }
}

impl<S: Space> Space for OptionSpace<S> {
    type Element = Option<S::Element>;

    fn contains(&self, value: &Self::Element) -> bool {
        match value {
            None => true,
            Some(inner_value) => self.inner.contains(inner_value),
        }
    }
}

impl<S: FiniteSpace> FiniteSpace for OptionSpace<S> {
    fn size(&self) -> usize {
        1 + self.inner.size()
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        match element {
            None => 0,
            Some(inner_elem) => 1 + self.inner.to_index(inner_elem),
        }
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        if index == 0 {
            Some(None)
        } else {
            Some(Some(self.inner.from_index(index - 1)?))
        }
    }
}

impl<S: NumFeatures> NumFeatures for OptionSpace<S> {
    fn num_features(&self) -> usize {
        1 + self.inner.num_features()
    }
}

impl<S: EncoderFeatureSpace> EncoderFeatureSpace for OptionSpace<S> {
    type Encoder = S::Encoder;

    fn encoder(&self) -> Self::Encoder {
        self.inner.encoder()
    }

    fn encoder_features_out<F: Float>(
        &self,
        element: &Self::Element,
        out: &mut [F],
        zeroed: bool,
        encoder: &Self::Encoder,
    ) {
        match element {
            None => {
                out[0] = F::one();
                if !zeroed {
                    out[1..].fill(F::zero())
                }
            }
            Some(inner_elem) => {
                out[0] = F::zero();
                self.inner
                    .encoder_features_out(inner_elem, &mut out[1..], zeroed, encoder)
            }
        }
    }
}

impl<S> Distribution<<Self as Space>::Element> for OptionSpace<S>
where
    S: Space + Distribution<S::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        // Sample None half of the time.
        if rng.gen() {
            None
        } else {
            Some(self.inner.sample(rng))
        }
    }
}

// NOTE: ElementRefTryInto instead of ElementRefInto?
impl<S: Space> ElementRefInto<Loggable> for OptionSpace<S> {
    fn elem_ref_into(&self, _element: &Self::Element) -> Loggable {
        // No clear way to convert structured elements into Loggable
        Loggable::Nothing
    }
}

#[cfg(test)]
mod space {
    use super::super::{testing, IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn contains_none() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert!(space.contains(&None));
    }

    #[test]
    fn contains_some() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert!(space.contains(&Some(())));
    }

    #[test]
    fn contains_samples_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_contains_samples(&space, 100);
    }

    #[test]
    fn contains_samples_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_contains_samples(&space, 100);
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::{testing, IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn from_to_index_iter_size_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_iter_size_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_index_sampled_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_index_sampled(&space, 10);
    }

    #[test]
    fn from_index_sampled_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_index_sampled(&space, 30);
    }

    #[test]
    fn from_index_invalid_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn from_index_invalid_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_index_invalid(&space);
    }
}

#[cfg(test)]
mod base_feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn num_features_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert_eq!(space.num_features(), 1);
    }

    #[test]
    fn num_features_index() {
        let space = OptionSpace::new(IndexSpace::new(3));
        assert_eq!(space.num_features(), 4);
    }
}

#[cfg(test)]
mod feature_space {
    use super::super::{FeatureSpace, IndexSpace, SingletonSpace};
    use super::*;
    use crate::utils::tensor::UniqueTensor;
    use ndarray::{arr1, Array};
    use tch::Tensor;

    macro_rules! features_tests {
        ($label:ident, $inner:expr, $elem:expr, $expected:expr) => {
            mod $label {
                use super::*;

                #[test]
                fn tensor_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Tensor = space.features(&$elem);
                    assert_eq!(actual, Tensor::of_slice(&$expected));
                }

                #[test]
                fn tensor_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected = Tensor::of_slice(&$expected);
                    let mut out = UniqueTensor::<f32, _>::zeros(expected.numel());
                    space.features_out(&$elem, out.as_slice_mut(), false);
                    assert_eq!(out.into_tensor(), expected);
                }

                #[test]
                fn array_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Array<f32, _> = space.features(&$elem);
                    let expected: Array<f32, _> = arr1(&$expected);
                    assert_eq!(actual, expected);
                }

                #[test]
                fn array_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected: Array<f32, _> = arr1(&$expected);
                    let mut out = Array::from_elem(expected.raw_dim(), f32::NAN);
                    space.features_out(&$elem, out.as_slice_mut().unwrap(), false);
                    assert_eq!(out, expected);
                }
            }
        };
    }

    features_tests!(singleton_none, SingletonSpace::new(), None, [1.0_f32]);
    features_tests!(singleton_some, SingletonSpace::new(), Some(()), [0.0_f32]);
    features_tests!(
        index_none,
        IndexSpace::new(3),
        None,
        [1.0, 0.0, 0.0, 0.0_f32]
    );
    features_tests!(
        index_some,
        IndexSpace::new(3),
        Some(1),
        [0.0, 0.0, 1.0, 0.0_f32]
    );
}

#[cfg(test)]
mod batch_feature_space {
    use super::super::{FeatureSpace, IndexSpace, SingletonSpace};
    use super::*;
    use crate::utils::tensor::UniqueTensor;
    use ndarray::{arr2, Array};
    use tch::Tensor;

    fn tensor_from_arrays<T: tch::kind::Element, const N: usize, const M: usize>(
        data: [[T; M]; N],
    ) -> Tensor {
        let flat_data: Vec<T> = data
            .into_iter()
            .map(IntoIterator::into_iter)
            .flatten()
            .collect();
        Tensor::of_slice(&flat_data).reshape(&[N as i64, M as i64])
    }

    macro_rules! batch_features_tests {
        ($label:ident, $inner:expr, $elems:expr, $expected:expr) => {
            mod $label {
                use super::*;

                #[test]
                fn tensor_batch_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Tensor = space.batch_features(&$elems);
                    assert_eq!(actual, tensor_from_arrays($expected));
                }

                #[test]
                fn tensor_batch_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected = tensor_from_arrays($expected);
                    let (a, b) = expected.size2().unwrap();
                    let mut out = UniqueTensor::<f32, _>::zeros((a as _, b as _));
                    space.batch_features_out(&$elems, &mut out.array_view_mut(), false);
                    assert_eq!(out.into_tensor(), expected);
                }

                #[test]
                fn array_batch_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Array<f32, _> = space.batch_features(&$elems);
                    let expected: Array<f32, _> = arr2(&$expected);
                    assert_eq!(actual, expected);
                }

                #[test]
                fn array_batch_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected: Array<f32, _> = arr2(&$expected);
                    let mut out = Array::from_elem(expected.raw_dim(), f32::NAN);
                    space.batch_features_out(&$elems, &mut out, false);
                    assert_eq!(out, expected);
                }
            }
        };
    }

    batch_features_tests!(
        singleton,
        SingletonSpace::new(),
        [Some(()), None, Some(())],
        [[0.0], [1.0], [0.0_f32]]
    );
    batch_features_tests!(
        index,
        IndexSpace::new(3),
        [Some(1), None, Some(0), Some(2), None],
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0_f32]
        ]
    );
}
