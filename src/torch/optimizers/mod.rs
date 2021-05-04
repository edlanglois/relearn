//! Optimizers
mod coptimizer;

pub use coptimizer::{AdamConfig, AdamWConfig, RmsPropConfig, SgdConfig};

use std::error::Error;
use tch::{nn::VarStore, Tensor};

/// Base optimizer interface
pub trait BaseOptimizer {
    /// Method error type
    type Error: Error;

    /// Zero out the gradients of all optimized tensors
    fn f_zero_grad(&self) -> Result<(), Self::Error>;

    /// Zero out the gradients of all optimized tensors
    ///
    /// # Panics
    /// This wraps [BaseOptimizer::f_zero_grad] and panics if `f_zero_grad` fails.
    fn zero_grad(&self) {
        self.f_zero_grad().unwrap();
    }
}

/// Optimizer that minimizes a loss function.
pub trait Optimizer: BaseOptimizer {
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
    ///         Used by some optimizers that require multiple evaluations of the loss.
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

/// Optimizer that minimizes a loss tensor.
pub trait OnceOptimizer: BaseOptimizer {
    /// Perform an optimization step (parameter update).
    ///
    /// See [OnceOptimizer::step_once].
    fn f_step_once(&self) -> Result<(), Self::Error>;

    /// Perform an optimization step (parameter update).
    ///
    /// Uses the existing gradients stored with the parameter tensor.
    ///
    /// # Panics
    /// This wraps [Optimizer::f_step_once] and panics if `f_step_once` fails.
    fn step_once(&self) {
        self.f_step_once().unwrap()
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

impl<T: OnceOptimizer> Optimizer for T {
    fn f_step(&self, _loss_fn: &dyn Fn() -> Tensor) -> Result<(), Self::Error> {
        self.f_step_once()
    }

    fn f_backward_step(&self, loss_fn: &dyn Fn() -> Tensor) -> Result<Tensor, Self::Error> {
        let loss = loss_fn();
        self.f_backward_step_once(&loss)?;
        Ok(loss)
    }
}

/// Build an optimizer
pub trait OptimizerBuilder<T> {
    type Error: Error;

    /// Build an optimizer for the trainable variables in a variable store.
    fn build_optimizer(&self, vs: &VarStore) -> Result<T, Self::Error>;
}

#[cfg(test)]
mod testing {
    use super::*;
    use tch::Device;

    pub fn check_optimizes_quadratic<O, OB>(builder: &OB, num_steps: u64)
    where
        O: Optimizer,
        OB: OptimizerBuilder<O>,
    {
        // Minimize f(x) = 1/2*x'Mx + b'x
        // with M = [1  -1]  b = [ 2]
        //          [-1  2]      [-3]
        //
        // which is minimized at x = [-1  1]'
        let m = Tensor::of_slice(&[1.0f32, -1.0, -1.0, 2.0]).reshape(&[2, 2]);
        let b = Tensor::of_slice(&[2.0f32, -3.0]);

        let vs = VarStore::new(Device::Cpu);
        let x = vs.root().f_zeros("x", &[2]).unwrap();
        let optimizer = builder.build_optimizer(&vs).unwrap();

        let loss_fn = || m.mv(&x).dot(&x) / 2 + b.dot(&x);

        for _ in 0..num_steps {
            let _ = optimizer.backward_step(&loss_fn);
        }

        let expected = Tensor::of_slice(&[-1.0, 1.0]);
        assert!(
            f64::from((&x - &expected).norm()) < 1e-3,
            "expected: {:?}, actual: {:?}",
            expected,
            x
        );
    }
}
