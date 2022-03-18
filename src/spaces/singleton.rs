//! Singleton space definition.
use super::{ParameterizedDistributionSpace, ProductSpace, ReprSpace};
use crate::torch::distributions::DeterministicEmptyVec;
use serde::{Deserialize, Serialize};
use std::fmt;
use tch::{Device, Kind, Tensor};

/// A space containing a single element.
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    ProductSpace,
    FiniteSpace,
    Serialize,
    Deserialize,
)]
pub struct SingletonSpace;

impl SingletonSpace {
    #[inline]
    pub const fn new() -> Self {
        SingletonSpace
    }
}

impl fmt::Display for SingletonSpace {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SingletonSpace")
    }
}

/// Represent elements as an integer vector of length 0.
impl ReprSpace<Tensor> for SingletonSpace {
    #[inline]
    fn repr(&self, _element: &Self::Element) -> Tensor {
        Tensor::empty(&[0], (Kind::Int64, Device::Cpu))
    }

    #[inline]
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

/// "Parameterize" a deterministic distribution with no parameters.
impl ParameterizedDistributionSpace<Tensor> for SingletonSpace {
    type Distribution = DeterministicEmptyVec;

    #[inline]
    fn num_distribution_params(&self) -> usize {
        0
    }

    #[inline]
    fn sample_element(&self, _params: &Tensor) -> Self::Element {}

    #[inline]
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
