//! `BooleanSpace` definition
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, ElementRefInto, FeatureSpace,
    FeatureSpaceOut, FiniteSpace, ReprSpace, Space,
};
use crate::logging::Loggable;
use crate::utils::array::BasicArray;
use rand::distributions::Distribution;
use rand::Rng;
use std::fmt;
use tch::{Device, Kind, Tensor};

/// A space containing a boolean value.
#[derive(Debug, Clone)]
pub struct BooleanSpace;

impl BooleanSpace {
    pub const fn new() -> Self {
        BooleanSpace
    }
}

impl Default for BooleanSpace {
    fn default() -> Self {
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

/// Represent an elements as a Boolean valued tensor.
impl ReprSpace<Tensor> for BooleanSpace {
    fn repr(&self, element: &Self::Element) -> Tensor {
        Tensor::scalar_tensor(*element as i64, (Kind::Bool, Device::Cpu))
    }

    fn batch_repr<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let elements: Vec<_> = elements.into_iter().cloned().collect();
        Tensor::of_slice(&elements)
    }
}

impl BaseFeatureSpace for BooleanSpace {
    fn num_features(&self) -> usize {
        1
    }
}

impl<T> FeatureSpace<T> for BooleanSpace
where
    T: BasicArray<f32, 1>,
{
    fn features(&self, element: &Self::Element) -> T {
        if *element {
            T::ones([1])
        } else {
            T::zeros([1])
        }
    }
}

impl FeatureSpaceOut<Tensor> for BooleanSpace {
    fn features_out(&self, element: &Self::Element, out: &mut Tensor, zeroed: bool) {
        if *element {
            let _ = out.fill_(1.0);
        } else if !zeroed {
            let _ = out.zero_();
        }
    }
}

impl BatchFeatureSpace<Tensor> for BooleanSpace {
    fn batch_features<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let elements: Vec<f64> = elements.into_iter().map(|&x| f64::from(x as u8)).collect();
        Tensor::of_slice(&elements).unsqueeze_(-1)
    }
}

impl BatchFeatureSpaceOut<Tensor> for BooleanSpace {
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut Tensor, _zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let elements: Vec<f64> = elements.into_iter().map(|&x| f64::from(x as u8)).collect();
        let _ = out.copy_(&Tensor::of_slice(&elements).unsqueeze_(-1));
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
    use super::*;
    use std::array::IntoIter;

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
                    let mut out = expected.empty_like();
                    space.features_out(&$elem, &mut out, false);
                    assert_eq!(out, expected);
                }
            }
        };
    }

    features_tests!(false_, false, [0.0]);
    features_tests!(true_, true, [1.0]);

    fn tensor_from_arrays<const N: usize, const M: usize>(data: [[f32; M]; N]) -> Tensor {
        let flat_data: Vec<f32> = IntoIter::new(data).map(IntoIter::new).flatten().collect();
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
        let mut out = expected.empty_like();
        space.batch_features_out(&[false, true, true, false], &mut out, false);
        assert_eq!(out, expected);
    }
}
