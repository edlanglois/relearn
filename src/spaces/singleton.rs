//! Singleton space definition.
use super::{
    ElementRefInto, EncoderFeatureSpace, FiniteSpace, NumFeatures, ParameterizedDistributionSpace,
    ReprSpace, Space,
};
use crate::logging::Loggable;
use crate::torch::distributions::DeterministicEmptyVec;
use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
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
mod partial_ord {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn eq() {
        assert_eq!(SingletonSpace::new(), SingletonSpace::new());
    }

    #[test]
    fn cmp_equal() {
        assert_eq!(
            SingletonSpace::new().cmp(&SingletonSpace::new()),
            Ordering::Equal
        );
    }

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn not_less() {
        assert!(!(SingletonSpace::new() < SingletonSpace::new()));
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
    use super::super::FeatureSpace;
    use super::*;
    use crate::utils::tensor::UniqueTensor;
    use ndarray::Array;
    use tch::Tensor;

    #[test]
    fn num_features() {
        let space = SingletonSpace::new();
        assert_eq!(space.num_features(), 0);
    }

    #[test]
    fn features_tensor() {
        let space = SingletonSpace::new();
        let actual: Tensor = space.features(&());
        assert_eq!(actual, Tensor::zeros(&[0], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn features_array() {
        let space = SingletonSpace::new();
        let actual: Array<f32, _> = space.features(&());
        assert_eq!(actual, Array::<f32, _>::zeros([0]));
    }

    #[test]
    fn batch_features_tensor() {
        let space = SingletonSpace::new();
        let actual: Tensor = space.batch_features(&[(), (), ()]);
        assert_eq!(actual, Tensor::zeros(&[3, 0], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn batch_features_array() {
        let space = SingletonSpace::new();
        let actual: Array<f32, _> = space.batch_features(&[(), (), ()]);
        assert_eq!(actual, Array::<f32, _>::zeros([3, 0]));
    }

    // The _out tests really just check that it doesn't panic since the array is empty.
    #[test]
    fn features_out_zeros_tensor() {
        let space = SingletonSpace::new();
        let mut out = UniqueTensor::<f32, _>::zeros([0]);
        space.features_out(&(), out.as_slice_mut(), true);
        assert_eq!(
            out.into_tensor(),
            Tensor::zeros(&[0], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn features_out_ones_tensor() {
        let space = SingletonSpace::new();
        let mut out = UniqueTensor::<f32, _>::zeros([0]);
        space.features_out(&(), out.as_slice_mut(), false);
        assert_eq!(
            out.into_tensor(),
            Tensor::zeros(&[0], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn features_out_array() {
        let space = SingletonSpace::new();
        let mut out = Array::from_elem([0], f32::NAN);
        space.features_out(&(), out.as_slice_mut().unwrap(), false);
        assert_eq!(out, Array::zeros([0]));
    }

    #[test]
    fn features_out_zeroed_array() {
        let space = SingletonSpace::new();
        let mut out: Array<f32, _> = Array::zeros([0]);
        space.features_out(&(), out.as_slice_mut().unwrap(), false);
        assert_eq!(out, Array::zeros([0]));
    }

    #[test]
    fn batch_features_out_tensor() {
        let space = SingletonSpace::new();
        let mut out = UniqueTensor::<f32, _>::zeros((3, 0));
        space.batch_features_out(&[(), (), ()], &mut out.array_view_mut(), false);
        assert_eq!(
            out.into_tensor(),
            Tensor::zeros(&[3, 0], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn batch_features_out_array() {
        let space = SingletonSpace::new();
        let mut out = Array::from_elem([3, 0], f32::NAN);
        space.batch_features_out(&[(), (), ()], &mut out, false);
        assert_eq!(out, Array::zeros([3, 0]));
    }
}
