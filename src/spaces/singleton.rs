//! Singleton space definition.
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, ElementRefInto, FeatureSpace,
    FeatureSpaceOut, FiniteSpace, ParameterizedDistributionSpace, ReprSpace, Space,
};
use crate::logging::Loggable;
use crate::torch::distributions::DeterministicEmptyVec;
use crate::utils::array::{BasicArray, BasicArrayMut};
use rand::distributions::Distribution;
use rand::Rng;
use std::fmt;
use tch::{Device, Kind, Tensor};

/// A space containing a single element.
#[derive(Debug, Clone)]
pub struct SingletonSpace;

impl SingletonSpace {
    pub const fn new() -> Self {
        SingletonSpace
    }
}

impl Default for SingletonSpace {
    fn default() -> Self {
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
        Self::Element: 'a,
    {
        let num_elements = elements.into_iter().count();
        Tensor::empty(&[num_elements as i64, 0], (Kind::Int64, Device::Cpu))
    }
}

impl BaseFeatureSpace for SingletonSpace {
    fn num_features(&self) -> usize {
        0
    }
}

/// Represent elements as a float vector of length 0.
impl<T> FeatureSpace<T> for SingletonSpace
where
    T: BasicArray<f32, 1>,
{
    fn features(&self, _element: &Self::Element) -> T {
        T::zeros([0])
    }
}

impl<T2> BatchFeatureSpace<T2> for SingletonSpace
where
    T2: BasicArray<f32, 2>,
{
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let num_elements = elements.into_iter().count();
        T2::zeros([num_elements, 0])
    }
}

impl<T> FeatureSpaceOut<T> for SingletonSpace
where
    T: BasicArrayMut,
{
    fn features_out(&self, _element: &Self::Element, _out: &mut T, _zeroed: bool) {}
}

impl<T2> BatchFeatureSpaceOut<T2> for SingletonSpace
where
    T2: BasicArrayMut,
{
    fn batch_features_out<'a, I>(&self, _elements: I, _out: &mut T2, _zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
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
        assert_eq!(actual, Array::zeros([0]));
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
        assert_eq!(actual, Array::zeros([3, 0]));
    }

    // The _out tests really just check that it doesn't panic since the array is empty.
    #[test]
    fn features_out_tensor() {
        let space = SingletonSpace::new();
        let mut out = Tensor::empty(&[0], (Kind::Float, Device::Cpu));
        space.features_out(&(), &mut out, false);
        assert_eq!(out, Tensor::zeros(&[0], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn features_out_zeroed_tensor() {
        let space = SingletonSpace::new();
        let mut out = Tensor::zeros(&[0], (Kind::Float, Device::Cpu));
        space.features_out(&(), &mut out, true);
        assert_eq!(out, Tensor::zeros(&[0], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn features_out_array() {
        let space = SingletonSpace::new();
        let mut out = Array::from_elem([0], f32::NAN);
        space.features_out(&(), &mut out, false);
        assert_eq!(out, Array::zeros([0]));
    }

    #[test]
    fn features_out_zeroed_array() {
        let space = SingletonSpace::new();
        let mut out: Array<f32, _> = Array::zeros([0]);
        space.features_out(&(), &mut out, false);
        assert_eq!(out, Array::zeros([0]));
    }

    #[test]
    fn batch_features_out_tensor() {
        let space = SingletonSpace::new();
        let mut out = Tensor::empty(&[3, 0], (Kind::Float, Device::Cpu));
        space.batch_features_out(&[(), (), ()], &mut out, false);
        assert_eq!(out, Tensor::zeros(&[3, 0], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn batch_features_out_array() {
        let space = SingletonSpace::new();
        let mut out = Array::from_elem([3, 0], f32::NAN);
        space.batch_features_out(&[(), (), ()], &mut out, false);
        assert_eq!(out, Array::zeros([3, 0]));
    }
}
