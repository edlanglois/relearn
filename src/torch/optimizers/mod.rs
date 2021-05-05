//! Optimizers
mod coptimizer;

pub use coptimizer::{AdamConfig, AdamWConfig, RmsPropConfig, SgdConfig};

use std::error::Error;
use tch::{nn::VarStore, Tensor};
use thiserror::Error;

/// Base optimizer interface
pub trait BaseOptimizer {
    /// Zero out the gradients of all optimized tensors
    fn zero_grad(&self);
}

/// Optimizer that minimizes a loss function.
pub trait Optimizer: BaseOptimizer {
    /// Apply an optimization step using the gradient of a loss function.
    ///
    /// Obtains gradients by backpropagating the result of `loss_fn`.
    ///
    /// # Args
    /// * `loss_fn` - Loss function.
    ///     Called to obtain the loss tensor, which is back-propagated to obtain a gradient.
    ///     Always evaluated at least once; may be evaluated multiple times.
    ///
    /// # Returns
    /// The initial value of `loss_fn` on success.
    ///
    /// If an error is detected, the parameters guaranteed to be unchanged from (or reset to)
    /// their initial values.
    /// In general, error conditions are not guaranteed to be detected and an optimizer
    /// may silently put itself or the parameters into a bad state.
    /// For example, [COptimizer] sets parameters to NaN when the loss is NaN.
    fn backward_step(&self, loss_fn: &dyn Fn() -> Tensor) -> Result<Tensor, OptimizerStepError>;
}

/// Optimizer that minimizes a loss tensor.
pub trait OnceOptimizer: BaseOptimizer {
    /// Perform an optimization step (parameter update).
    ///
    /// Uses the existing gradients stored with the parameter tensor.
    ///
    /// If an error is detected, the parameters guaranteed to be unchanged from (or reset to)
    /// their initial values.
    /// In general, error conditions are not guaranteed to be detected and an optimizer
    /// may silently put itself or the parameters into a bad state.
    /// For example, [COptimizer] sets parameters to NaN when the loss is NaN.
    fn step_once(&self) -> Result<(), OptimizerStepError>;

    /// Apply a backward step pass, update the gradients, and perform an optimization step.
    ///
    /// Obtains gradients by backpropagating the result of `loss_fn`.
    ///
    /// # Args
    /// * `loss` - Loss tensor. Back-propagation is applied to this tensor to obtain a gradient.
    ///
    /// # Returns
    /// The initial value of `loss_fn` on success.
    ///
    /// If an error is detected, the parameters guaranteed to be unchanged from (or reset to)
    /// their initial values.
    /// In general, error conditions are not guaranteed to be detected and an optimizer
    /// may silently put itself or the parameters into a bad state.
    /// For example, [COptimizer] sets parameters to NaN when the loss is NaN.
    fn backward_step_once(&self, loss: &Tensor) -> Result<(), OptimizerStepError>;
}

impl<T: OnceOptimizer> Optimizer for T {
    fn backward_step(&self, loss_fn: &dyn Fn() -> Tensor) -> Result<Tensor, OptimizerStepError> {
        let loss = loss_fn();
        self.backward_step_once(&loss)?;
        Ok(loss)
    }
}

/// Error performing an optimization step.
#[derive(Debug, Error)]
pub enum OptimizerStepError {}

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
