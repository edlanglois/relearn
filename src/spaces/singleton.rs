//! Singleton space definition.
use super::{
    BaseFeatureSpace, ElementRefInto, FeatureSpace, FeatureSpaceOut, FiniteSpace,
    ParameterizedDistributionSpace, ReprSpace, Space,
};
use crate::logging::Loggable;
use crate::torch::distributions::DeterministicEmptyVec;
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
impl FeatureSpace<Tensor> for SingletonSpace {
    fn features(&self, _element: &Self::Element) -> Tensor {
        Tensor::empty(&[0], (Kind::Float, Device::Cpu))
    }

    fn batch_features<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let num_elements = elements.into_iter().count();
        Tensor::empty(&[num_elements as i64, 0], (Kind::Int64, Device::Cpu))
    }
}

impl FeatureSpaceOut<Tensor> for SingletonSpace {
    fn features_out(&self, _element: &Self::Element, _out: &mut Tensor) {}

    fn batch_features_out<'a, I>(&self, _elements: I, _out: &mut Tensor)
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
