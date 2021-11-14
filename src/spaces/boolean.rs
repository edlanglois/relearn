//! `BooleanSpace` definition
use super::{ElementRefInto, EncoderFeatureSpace, FiniteSpace, NumFeatures, ReprSpace, Space};
use crate::logging::Loggable;
use crate::utils::num_array::{BuildFromArray1D, NumArray1D};
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use std::cmp::Ordering;
use std::fmt;
use tch::{Device, Kind, Tensor};

/// The space `{false, true}`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BooleanSpace;

impl BooleanSpace {
    pub const fn new() -> Self {
        BooleanSpace
    }
}

impl fmt::Display for BooleanSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BooleanSpace")
    }
}

impl Space for BooleanSpace {
    type Element = bool;

    fn contains(&self, _value: &Self::Element) -> bool {
        true
    }
}

impl PartialOrd for BooleanSpace {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BooleanSpace {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl FiniteSpace for BooleanSpace {
    fn size(&self) -> usize {
        2
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        (*element).into()
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        match index {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
        Some(index != 0)
    }
}

/// Represent elements as a Boolean valued tensor.
impl ReprSpace<Tensor> for BooleanSpace {
    fn repr(&self, element: &Self::Element) -> Tensor {
        Tensor::scalar_tensor(*element as i64, (Kind::Bool, Device::Cpu))
    }

    fn batch_repr<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        let elements: Vec<_> = elements.into_iter().cloned().collect();
        Tensor::of_slice(&elements)
    }
}

impl NumFeatures for BooleanSpace {
    fn num_features(&self) -> usize {
        1
    }
}

impl EncoderFeatureSpace for BooleanSpace {
    type Encoder = ();
    fn encoder(&self) -> Self::Encoder {}
    fn encoder_features_out<F: Float>(
        &self,
        element: &Self::Element,
        out: &mut [F],
        _zeroed: bool,
        _encoder: &Self::Encoder,
    ) {
        out[0] = if *element { F::one() } else { F::zero() };
    }

    fn encoder_features<T>(&self, element: &Self::Element, _encoder: &Self::Encoder) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        if *element {
            T::Array::ones(1).into()
        } else {
            T::Array::zeros(1).into()
        }
    }
}

impl Distribution<<Self as Space>::Element> for BooleanSpace {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        rng.gen()
    }
}

impl ElementRefInto<Loggable> for BooleanSpace {
    fn elem_ref_into(&self, element: &Self::Element) -> Loggable {
        Loggable::Scalar((*element as u8).into())
    }
}

#[cfg(test)]
mod space {
    use super::super::testing;
    use super::*;

    #[test]
    fn contains_false() {
        let space = BooleanSpace::new();
        assert!(space.contains(&false));
    }

    #[test]
    fn contains_true() {
        let space = BooleanSpace::new();
        assert!(space.contains(&true));
    }

    #[test]
    fn contains_samples() {
        let space = BooleanSpace::new();
        testing::check_contains_samples(&space, 10);
    }
}

#[cfg(test)]
mod partial_ord {
    use super::*;

    #[test]
    fn eq() {
        assert_eq!(BooleanSpace::new(), BooleanSpace::new());
    }

    #[test]
    fn cmp_equal() {
        assert_eq!(
            BooleanSpace::new().cmp(&BooleanSpace::new()),
            Ordering::Equal
        );
    }

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn not_less() {
        assert!(!(BooleanSpace::new() < BooleanSpace::new()));
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::testing;
    use super::*;

    #[test]
    fn from_to_index_iter_size() {
        let space = BooleanSpace::new();
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_random() {
        let space = BooleanSpace::new();
        testing::check_from_to_index_random(&space, 10);
    }

    #[test]
    fn from_index_sampled() {
        let space = BooleanSpace::new();
        testing::check_from_index_sampled(&space, 10);
    }

    #[test]
    fn from_index_invalid() {
        let space = BooleanSpace::new();
        testing::check_from_index_invalid(&space);
    }
}

#[cfg(test)]
mod feature_space {
    use super::super::FeatureSpace;
    use super::*;
    use crate::utils::tensor::UniqueTensor;

    #[test]
    fn num_features() {
        let space = BooleanSpace::new();
        assert_eq!(space.num_features(), 1);
    }

    macro_rules! features_tests {
        ($label:ident, $elem:expr, $expected:expr) => {
            mod $label {
                use super::*;

                #[test]
                fn tensor_features() {
                    let space = BooleanSpace::new();
                    let actual: Tensor = space.features(&$elem);
                    let expected_vec: &[f32] = &$expected;
                    assert_eq!(actual, Tensor::of_slice(expected_vec));
                }

                #[test]
                fn tensor_features_out() {
                    let space = BooleanSpace::new();
                    let expected_vec: &[f32] = &$expected;
                    let expected = Tensor::of_slice(&expected_vec);
                    let mut out = UniqueTensor::<f32, _>::zeros(expected_vec.len());
                    space.features_out(&$elem, out.as_slice_mut(), true);
                    assert_eq!(out.into_tensor(), expected);
                }
            }
        };
    }

    features_tests!(false_, false, [0.0]);
    features_tests!(true_, true, [1.0]);

    fn tensor_from_arrays<const N: usize, const M: usize>(data: [[f32; M]; N]) -> Tensor {
        let flat_data: Vec<f32> = data
            .into_iter()
            .map(IntoIterator::into_iter)
            .flatten()
            .collect();
        Tensor::of_slice(&flat_data).reshape(&[N as i64, M as i64])
    }

    #[test]
    fn tensor_batch_features() {
        let space = BooleanSpace::new();
        let actual: Tensor = space.batch_features(&[false, true, true, false]);
        assert_eq!(actual, tensor_from_arrays([[0.0], [1.0], [1.0], [0.0]]));
    }

    #[test]
    fn tensor_batch_features_out() {
        let space = BooleanSpace::new();
        let expected = tensor_from_arrays([[0.0], [1.0], [1.0], [0.0]]);
        let mut out = UniqueTensor::<f32, _>::zeros((4, 1));
        space.batch_features_out(
            &[false, true, true, false],
            &mut out.array_view_mut(),
            false,
        );
        assert_eq!(out.into_tensor(), expected);
    }
}
