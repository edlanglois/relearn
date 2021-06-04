//! Activation functions
use crate::torch::backends::CudnnSupport;
use clap::Clap;
use tch::{nn, Tensor};

/// Activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Clap)]
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

impl Activation {
    /// Create a module encapsulating this function if not the identity.
    pub fn maybe_module(self) -> Option<nn::Func<'static>> {
        use Activation::*;
        match self {
            Identity => None,
            _ => Some(self.module()),
        }
    }
    /// Create a module encapsulating this function.
    pub fn module(self) -> nn::Func<'static> {
        use Activation::*;
        match self {
            Identity => nn::func(Tensor::shallow_clone),
            Relu => nn::func(Tensor::relu),
            Sigmoid => nn::func(Tensor::sigmoid),
            Tanh => nn::func(Tensor::tanh),
        }
    }
}

impl CudnnSupport for Activation {
    fn has_cudnn_second_derivatives(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod activation {
    use super::*;
    use rstest::rstest;

    #[test]
    fn identity_maybe_module_none() {
        assert!(Activation::Identity.maybe_module().is_none());
    }

    #[rstest]
    #[case(Activation::Relu)]
    #[case(Activation::Sigmoid)]
    #[case(Activation::Tanh)]
    fn maybe_module_some(#[case] activation: Activation) {
        assert!(activation.maybe_module().is_some());
    }

    #[test]
    fn module_identity() {
        let x = Tensor::of_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let activation_fn = Activation::Identity.module();
        assert_eq!(x.apply(&activation_fn), x);
    }

    #[test]
    fn module_relu() {
        let x = Tensor::of_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let activation_fn = Activation::Relu.module();
        let expected = Tensor::of_slice(&[0.0, 0.0, 0.0, 1.0, 2.0]);
        assert_eq!(x.apply(&activation_fn), expected);
    }

    #[rstest]
    #[case(Activation::Relu, 0.0, f64::INFINITY)]
    #[case(Activation::Sigmoid, 0.0, 1.0)]
    #[case(Activation::Tanh, -1.0, 1.0)]
    fn module_bounds(
        #[case] activation: Activation,
        #[case] lower_bound: f64,
        #[case] upper_bound: f64,
    ) {
        let x = Tensor::of_slice(&[f64::NEG_INFINITY, -2.0, -1.0, 0.0, 1.0, 2.0, f64::INFINITY]);
        let y = x.apply(&activation.module());

        assert!(bool::from(y.greater_equal(lower_bound).all()));
        assert!(bool::from(y.less_equal(upper_bound).all()));
    }
}
