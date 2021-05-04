//! Optimizers
mod coptimizer;

pub use coptimizer::{AdamConfig, AdamWConfig, RmsPropConfig, SgdConfig};

use std::error::Error;
use tch::{nn::VarStore, Tensor};

/// Torch optimizer interface
pub trait Optimizer {
    type Error: Error;

    /// Zero out the gradients of all optimized tensors
    fn f_zero_grad(&self) -> Result<(), Self::Error>;

    /// Zero out the gradients of all optimized tensors
    fn zero_grad(&self) {
        self.f_zero_grad().unwrap();
    }

    /// Perform a single optimization step (parameter update).
    ///
    /// See [Optimizer::step].
    fn f_step(&self, loss_fn: &dyn Fn() -> Tensor) -> Result<(), Self::Error>;

    /// Perform a single optimization step (parameter update).
    ///
    /// Uses the existing gradients stored with the parameter tensor.
    ///
    /// # Args
    /// * `loss_fn` - Forward loss function.
    ///         Used by some optimizers that require multiple evaluations of the loss,
    ///         like Conjugate Gradient.
    ///         No backpropagation is applied to the result of this function.
    ///
    /// # Panics
    /// This wraps [Optimizer::f_step] and panics if `f_step` fails.
    fn step(&self, loss_fn: &dyn Fn() -> Tensor) {
        self.f_step(loss_fn).unwrap()
    }

    /// Apply an optimization step using the gradient of a loss function.
    ///
    /// See [Optimizer::backward_step].
    fn f_backward_step(&self, loss_fn: &dyn Fn() -> Tensor) -> Result<Tensor, Self::Error>;

    /// Apply an optimization step using the gradient of a loss function.
    ///
    /// Obtains gradients by backpropagating the result of `loss_fn`.
    ///
    /// # Args
    /// * `loss_fn` - Loss function.
    ///     Called to obtain the loss tensor, which is back-propagated to obtain a gradient.
    ///     Always evaluated at least once, may be evaluated multiple times.
    ///
    /// # Returns
    /// The initial value of `loss_fn`.
    ///
    /// # Panics
    /// This wraps [Optimizer::f_backward_step] and panics if `f_backward_step` fails.
    fn backward_step(&self, loss_fn: &dyn Fn() -> Tensor) -> Tensor {
        self.f_backward_step(loss_fn).unwrap()
    }
}

/// Torch optimizer that only requires a single evaluation of the loss on each step.
pub trait OnceOptimizer: Optimizer {
    /// Perform a single optimization step (parameter update).
    ///
    /// See [OnceOptimizer::step_once].
    fn f_step_once(&self) -> Result<(), Self::Error>;

    /// Perform a single optimization step (parameter update).
    ///
    /// Uses the existing gradients stored with the parameter tensor.
    ///
    /// # Panics
    /// This wraps [OnceOptimizer::f_step_once] and panics if `f_step_once` fails.
    fn step_once(&self) {
        self.f_step_once().unwrap();
    }

    /// Apply a backward step pass, update the gradients, and perform an optimization step.
    ///
    /// See [OnceOptimizer::backward_step_once].
    fn f_backward_step_once(&self, loss: &Tensor) -> Result<(), Self::Error>;

    /// Apply a backward step pass, update the gradients, and perform an optimization step.
    ///
    /// Obtains gradients by backpropagating the result of `loss_fn`.
    ///
    /// # Args
    /// * `loss` - Loss tensor. Back-propagation is applied to this tensor to obtain a gradient.
    ///
    /// # Panics
    /// This wraps [OnceOptimizer::f_backward_step_once]
    /// and panics if `f_backward_step_once` fails.
    fn backward_step_once(&self, loss: &Tensor) {
        self.f_backward_step_once(loss).unwrap();
    }
}

/// Build an optimizer
pub trait OptimizerBuilder<T> {
    type Error: Error;

    /// Build an optimizer for the trainable variables in a variable store.
    fn build_optimizer(&self, vs: &VarStore) -> Result<T, Self::Error>;
}
