//! FiniteSpace trait definition
use super::{FeatureSpace, Space};
use crate::torch::utils as torch_utils;
use tch::{kind::Kind, Device, Tensor};

/// A space containing finitely many elements.
pub trait FiniteSpace: Space {
    /// The number of elements in the space.
    fn size(&self) -> usize;

    /// Get the index of an element.
    fn to_index(&self, element: &Self::Element) -> usize;

    /// Try to convert an index to an element.
    ///
    /// If None is returned then the index was invalid.
    /// It is allowed that Some value may be returned even if the index is invalid.
    /// If you need to validate the returned value, use contains().
    fn from_index(&self, index: usize) -> Option<Self::Element>;
}

impl<S: FiniteSpace> FeatureSpace<Tensor> for S {
    fn num_features(&self) -> usize {
        self.size()
    }

    fn features(&self, element: &Self::Element) -> Tensor {
        torch_utils::one_hot(
            &Tensor::scalar_tensor(self.to_index(&element) as i64, (Kind::Int64, Device::Cpu)),
            self.num_features(),
            Kind::Float,
        )
    }

    fn batch_features<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let indices: Vec<_> = elements
            .into_iter()
            .map(|element| self.to_index(element) as i64)
            .collect();
        torch_utils::one_hot(
            &Tensor::of_slice(&indices),
            self.num_features(),
            Kind::Float,
        )
    }
}
