//! Optimizers
mod conjugate_gradient;
mod coptimizer;

pub use conjugate_gradient::ConjugateGradientOptimizer;
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
    /// * `loss_fn` - Loss function to minimize.
    ///     Called to obtain the loss tensor, which is back-propagated to obtain a gradient.
    ///     Always evaluated at least once; may be evaluated multiple times.
    ///
    /// # Returns
    /// The initial value of `loss_fn` on success.
    ///
    /// If an error is detected, the parameters are guaranteed to be unchanged from (or reset to)
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
    /// If an error is detected, the parameters are guaranteed to be unchanged from (or reset to)
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
    /// If an error is detected, the parameters are guaranteed to be unchanged from (or reset to)
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

/// Optimizer that minimizes a loss function subject to a trust region constraint on each step.
pub trait TrustRegionOptimizer: BaseOptimizer {
    /// Obtains gradients by backpropagating the result of `loss_fn`.
    /// May also backpropagate the result of `mistrust_fn`.
    ///
    /// # Args
    /// * `loss_fn` - Loss function to minimize.
    ///     Called to obtain the loss tensor, which is back-propagated to obtain a gradient.
    ///     Always evaluated at least once, may be evaluated multiple times.
    ///
    /// * `distance_fn` - Function that measures the distance of the current parameter values from
    ///     the initial parameters.
    ///     * Called as `distance_fn(&initial_params)`.
    ///     * `initial_params` will always be a vectorized copy of the trainable parameter values
    ///         as they were at the start of the call to `trust_region_backward_step`.
    ///     * The current parameter values are set directly in the trainable parameter tensors.
    ///     * Must return a 1-element tensor that is >= 0.
    ///     * Should return 0 if the current parameters are the same as `initial_params`.
    ///
    /// * `max_distance` - Upper bound on `distance_fn` for this step.
    ///
    /// # Returns
    /// The initial value of `loss_fn` on success.
    fn trust_region_backward_step(
        &self,
        loss_fn: &dyn Fn() -> Tensor,
        distance_fn: &dyn Fn() -> Tensor,
        max_distance: f64,
    ) -> Result<Tensor, OptimizerStepError>;
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
    use tch::{Device, IndexOp, Kind};

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
            let _ = optimizer.backward_step(&loss_fn).unwrap();
        }

        let expected = Tensor::of_slice(&[-1.0, 1.0]);
        assert!(
            f64::from((&x - &expected).norm()) < 1e-3,
            "expected: {:?}, actual: {:?}",
            expected,
            x
        );
    }

    pub fn check_trust_region_optimizes_quadratic<O, OB>(builder: &OB, num_steps: u64)
    where
        O: TrustRegionOptimizer,
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
        let x = vs.root().zeros("x", &[2]);
        let optimizer = builder.build_optimizer(&vs).unwrap();

        let loss_fn = || m.mv(&x).dot(&x) / 2 + b.dot(&x);

        let mut x_last = x.copy();

        for _ in 0..num_steps {
            let _ = x_last.copy_(&x.detach());
            let distance_fn = || (&x - &x_last).square().sum(Kind::Float);
            let _ = optimizer
                .trust_region_backward_step(&loss_fn, &distance_fn, 0.001)
                .unwrap();
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
