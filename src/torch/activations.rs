//! Activation functions
use tch::{nn, Tensor};

/// Activation functions.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Activation {
    /// No transformation
    Identity,
    /// Rectified linear
    Relu,
    /// Sigmoid function
    Sigmoid,
}

impl Activation {
    /// Create a module encapsulating this function if not the identity.
    pub fn maybe_module(&self) -> Option<nn::Func<'static>> {
        use Activation::*;
        match self {
            Identity => None,
            _ => Some(self.module()),
        }
    }
    /// Create a module encapsulating this function.
    pub fn module(&self) -> nn::Func<'static> {
        use Activation::*;
        match self {
            Identity => nn::func(|x| x.shallow_clone()),
            Relu => nn::func(Tensor::relu),
            Sigmoid => nn::func(Tensor::sigmoid),
        }
    }
}
