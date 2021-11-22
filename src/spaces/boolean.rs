//! `BooleanSpace` definition
use super::{
    ElementRefInto, EncoderFeatureSpace, FiniteSpace, NonEmptySpace, NumFeatures, ReprSpace, Space,
    SubsetOrd,
};
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

impl SubsetOrd for BooleanSpace {
    fn subset_cmp(&self, _other: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
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

impl NonEmptySpace for BooleanSpace {
    fn some_element(&self) -> Self::Element {
        false
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
mod subset_ord {
    use super::*;

    #[test]
    fn eq() {
        assert_eq!(BooleanSpace::new(), BooleanSpace::new());
    }

    #[test]
    fn cmp_equal() {
        assert_eq!(
            BooleanSpace::new().subset_cmp(&BooleanSpace::new()),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn not_less() {
        assert!(!BooleanSpace::new().strict_subset_of(&BooleanSpace::new()));
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

    const fn space() -> BooleanSpace {
        BooleanSpace::new()
    }

    #[test]
    fn num_features() {
        let space = space();
        assert_eq!(space.num_features(), 1);
    }

    features_tests!(false_, space(), false, [0.0]);
    features_tests!(true_, space(), true, [1.0]);
    batch_features_tests!(
        batch,
        space(),
        [false, true, true, false],
        [[0.0], [1.0], [1.0], [0.0]]
    );
}
