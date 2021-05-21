//! Wrap spaces to have non-empty feature vectors.
use super::{FeatureSpace, FiniteSpace, Space};
use std::fmt;
use tch::{Device, Kind, Tensor};

// Note: This could be renamed to something like TorchSpace if other helper changes end up being
// requried to satisfy torch methods.

/// Wrapper space with a feature vector length of at least 1.
///
/// Any generated features have value `0`.
#[derive(Debug, Clone)]
pub struct NonEmptyFeatures<S> {
    inner: S,
}

impl<S> NonEmptyFeatures<S> {
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: fmt::Display> fmt::Display for NonEmptyFeatures<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NonEmptyFeatures<{}>", self.inner)
    }
}

impl<S: Space> Space for NonEmptyFeatures<S> {
    type Element = S::Element;

    fn contains(&self, value: &Self::Element) -> bool {
        self.inner.contains(value)
    }
}

impl<S: FiniteSpace> FiniteSpace for NonEmptyFeatures<S> {
    fn size(&self) -> usize {
        self.inner.size()
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        self.inner.to_index(element)
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        self.inner.from_index(index)
    }
}

impl<S: FeatureSpace<Tensor>> FeatureSpace<Tensor> for NonEmptyFeatures<S> {
    fn num_features(&self) -> usize {
        self.inner.num_features().max(1)
    }

    fn features(&self, element: &Self::Element) -> Tensor {
        if self.inner.num_features() == 0 {
            Tensor::zeros(&[1], (Kind::Float, Device::Cpu))
        } else {
            self.inner.features(element)
        }
    }

    fn batch_features<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        if self.inner.num_features() == 0 {
            let num_elements = elements.into_iter().count();
            Tensor::zeros(&[num_elements as i64, 1], (Kind::Float, Device::Cpu))
        } else {
            self.inner.batch_features(elements)
        }
    }
}
