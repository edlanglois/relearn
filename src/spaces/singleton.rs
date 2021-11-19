//! Singleton space definition.
use super::{
    ElementRefInto, EncoderFeatureSpace, FiniteSpace, NumFeatures, ParameterizedDistributionSpace,
    ReprSpace, Space, SubsetOrd,
};
use crate::logging::Loggable;
use crate::torch::distributions::DeterministicEmptyVec;
use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use std::cmp::Ordering;
use std::fmt;
use tch::{Device, Kind, Tensor};

/// A space containing a single element.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SingletonSpace;

impl SingletonSpace {
    pub const fn new() -> Self {
        SingletonSpace
    }
}

impl fmt::Display for SingletonSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SingletonSpace")
    }
}

impl Space for SingletonSpace {
    type Element = ();

    fn contains(&self, _value: &Self::Element) -> bool {
        true
    }
}

impl SubsetOrd for SingletonSpace {
    fn subset_cmp(&self, _other: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl FiniteSpace for SingletonSpace {
    fn size(&self) -> usize {
        1
    }

    fn to_index(&self, _element: &Self::Element) -> usize {
        0
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        if index == 0 {
            Some(())
        } else {
            None
        }
    }

    fn from_index_unchecked(&self, _index: usize) -> Option<Self::Element> {
        Some(())
    }
}

/// Represent elements as an integer vector of length 0.
impl ReprSpace<Tensor> for SingletonSpace {
    fn repr(&self, _element: &Self::Element) -> Tensor {
        Tensor::empty(&[0], (Kind::Int64, Device::Cpu))
    }

    fn batch_repr<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        let num_elements = elements.into_iter().len();
        Tensor::empty(&[num_elements as i64, 0], (Kind::Int64, Device::Cpu))
    }
}

impl NumFeatures for SingletonSpace {
    fn num_features(&self) -> usize {
        0
    }
}

// Encode elements as length-0 vectors
impl EncoderFeatureSpace for SingletonSpace {
    type Encoder = ();
    fn encoder(&self) -> Self::Encoder {}

    #[inline(always)]
    fn encoder_features_out<F: Float>(
        &self,
        _element: &Self::Element,
        _out: &mut [F],
        _zeroed: bool,
        _encoder: &Self::Encoder,
    ) {
    }

    fn encoder_batch_features_out<'a, I, A>(
        &self,
        _elements: I,
        _out: &mut ArrayBase<A, Ix2>,
        _zeroed: bool,
        _encoder: &Self::Encoder,
    ) where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
    }
}

/// "Parameterize" a deterministic distribution with no parameters.
impl ParameterizedDistributionSpace<Tensor> for SingletonSpace {
    type Distribution = DeterministicEmptyVec;

    fn num_distribution_params(&self) -> usize {
        0
    }
    fn sample_element(&self, _params: &Tensor) -> Self::Element {}

    fn distribution(&self, params: &Tensor) -> Self::Distribution {
        let batch_shape: Vec<_> = params
            .size()
            .split_last()
            .expect("params must have shape [BATCH_SHAPE..., 0]")
            .1
            .into();
        Self::Distribution::new(batch_shape)
    }
}

impl Distribution<<Self as Space>::Element> for SingletonSpace {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> <Self as Space>::Element {}
}

impl ElementRefInto<Loggable> for SingletonSpace {
    fn elem_ref_into(&self, _element: &Self::Element) -> Loggable {
        Loggable::Nothing
    }
}

#[cfg(test)]
mod space {
    use super::super::testing;
    use super::*;

    #[test]
    fn contains_unit() {
        let space = SingletonSpace::new();
        assert!(space.contains(&()));
    }

    #[test]
    fn contains_samples() {
        let space = SingletonSpace::new();
        testing::check_contains_samples(&space, 10);
    }
}

#[cfg(test)]
mod subset_ord {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn eq() {
        assert_eq!(SingletonSpace::new(), SingletonSpace::new());
    }

    #[test]
    fn cmp_equal() {
        assert_eq!(
            SingletonSpace::new().subset_cmp(&SingletonSpace::new()),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn not_strict_subset() {
        assert!(!SingletonSpace::new().strict_subset_of(&SingletonSpace::new()));
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::testing;
    use super::*;

    #[test]
    fn from_to_index_iter_size() {
        let space = SingletonSpace::new();
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_random() {
        let space = SingletonSpace::new();
        testing::check_from_to_index_random(&space, 10);
    }

    #[test]
    fn from_index_sampled() {
        let space = SingletonSpace::new();
        testing::check_from_index_sampled(&space, 10);
    }

    #[test]
    fn from_index_invalid() {
        let space = SingletonSpace::new();
        testing::check_from_index_invalid(&space);
    }
}

#[cfg(test)]
mod feature_space {
    use super::*;

    #[test]
    fn num_features() {
        let space = SingletonSpace::new();
        assert_eq!(space.num_features(), 0);
    }

    features_tests!(f, SingletonSpace::new(), (), []);
    batch_features_tests!(b, SingletonSpace::new(), [(), (), ()], [[], [], []]);
}
