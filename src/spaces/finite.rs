//! FiniteSpace trait definition
use super::{FeatureSpace, ParameterizedSampleSpace, Space};
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

impl<S: FiniteSpace> ParameterizedSampleSpace<Tensor> for S {
    fn num_sample_params(&self) -> usize {
        self.size()
    }

    fn sample(&self, parameters: &Tensor) -> Self::Element {
        self.from_index(
            Into::<i64>::into(parameters.softmax(-1, Kind::Float).multinomial(1, true)) as usize,
        )
        .unwrap()
    }

    fn batch_log_probs<'a, I>(&self, parameters: &Tensor, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let indices: Vec<_> = elements
            .into_iter()
            .map(|element| self.to_index(element) as i64)
            .collect();
        let index_tensor = Tensor::of_slice(&indices);
        let logits = parameters.log_softmax(-1, Kind::Float);
        logits
            .gather(-1, &index_tensor.unsqueeze(-1), false)
            .squeeze1(-1)
    }
}
