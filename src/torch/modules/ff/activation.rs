//! Activation functions.
use super::super::{FeedForwardModule, IterativeModule, Module, ModuleExtras, SequenceModule};
use crate::torch::packed::PackedTensor;
use serde::{Deserialize, Serialize};
use std::iter;
use tch::{Device, Tensor};

/// Activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    /// No transformation
    Identity,
    /// Rectified linear
    Relu,
    /// Sigmoid function
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
}

impl Default for Activation {
    #[inline]
    fn default() -> Self {
        Self::Relu
    }
}

impl Activation {
    /// Apply to an owned tensor
    #[inline]
    pub fn forward_owned(&self, tensor: Tensor) -> Tensor {
        match self {
            Self::Identity => tensor,
            _ => self.forward(&tensor),
        }
    }
}

impl Module for Activation {
    #[inline]
    fn shallow_clone(&self) -> Self
    where
        Self: Sized,
    {
        *self
    }

    #[inline]
    fn clone_to_device(&self, _: Device) -> Self
    where
        Self: Sized,
    {
        *self
    }

    #[inline]
    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(ModuleExtras::variables(self))
    }

    #[inline]
    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(ModuleExtras::trainable_variables(self))
    }
}

impl<'a> ModuleExtras<'a> for Activation {
    type Variables = iter::Empty<&'a Tensor>;
    type TrainableVariables = Self::Variables;

    #[inline]
    fn variables(&'a self) -> Self::Variables {
        iter::empty()
    }

    #[inline]
    fn trainable_variables(&'a self) -> Self::TrainableVariables {
        iter::empty()
    }
}

impl FeedForwardModule for Activation {
    #[inline]
    fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Self::Identity => input.shallow_clone(),
            Self::Relu => input.relu(),
            Self::Sigmoid => input.sigmoid(),
            Self::Tanh => input.tanh(),
        }
    }
}

/// Sequence processing by batching over the sequence dimension.
impl SequenceModule for Activation {
    #[inline]
    fn seq_serial(&self, inputs: &Tensor, _seq_lengths: &[usize]) -> Tensor {
        self.forward(inputs)
    }

    #[inline]
    fn seq_packed(&self, inputs: &PackedTensor) -> PackedTensor {
        inputs.batch_map_ref(|tensor| self.forward(tensor))
    }
}

/// Iterate over a sequence by independently and identically transforming each step.
impl IterativeModule for Activation {
    type State = ();

    #[inline]
    fn initial_state(&self) -> Self::State {}

    #[inline]
    fn step(&self, _: &mut Self::State, input: &Tensor) -> Tensor {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn identity_forward() {
        let x = Tensor::of_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(Activation::Identity.forward(&x), x);
    }

    #[test]
    fn identity_forward_owned() {
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let x = Tensor::of_slice(&data);
        assert_eq!(
            Activation::Identity.forward_owned(x),
            Tensor::of_slice(&data)
        );
    }

    #[test]
    fn relu_forward() {
        let x = Tensor::of_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let expected = Tensor::of_slice(&[0.0, 0.0, 0.0, 1.0, 2.0]);
        assert_eq!(Activation::Relu.forward(&x), expected);
    }

    #[rstest]
    #[case(Activation::Relu, 0.0, f64::INFINITY)]
    #[case(Activation::Sigmoid, 0.0, 1.0)]
    #[case(Activation::Tanh, -1.0, 1.0)]
    fn forward_bounds(
        #[case] activation: Activation,
        #[case] lower_bound: f64,
        #[case] upper_bound: f64,
    ) {
        let x = Tensor::of_slice(&[f64::NEG_INFINITY, -2.0, -1.0, 0.0, 1.0, 2.0, f64::INFINITY]);
        let y = activation.forward(&x);

        assert!(bool::from(y.greater_equal(lower_bound).all()));
        assert!(bool::from(y.less_equal(upper_bound).all()));
    }

    #[test]
    fn variables_count() {
        assert_eq!(Module::variables(&Activation::Relu).count(), 0);
    }

    #[test]
    fn trainable_variables_count() {
        assert_eq!(Module::trainable_variables(&Activation::Relu).count(), 0);
    }
}
