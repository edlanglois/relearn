//! Optimizers
mod conjugate_gradient;
mod coptimizer;

pub use conjugate_gradient::{ConjugateGradientOptimizer, ConjugateGradientOptimizerConfig};
pub use coptimizer::{AdamConfig, AdamWConfig, RmsPropConfig, SgdConfig};

use crate::logging::Logger;
use std::error::Error;
use tch::{nn::VarStore, Tensor};
use thiserror::Error;

/// Base optimizer interface
pub trait BaseOptimizer {
    /// Zero out the gradients of all optimized tensors
    fn zero_grad(&mut self);
}

/// Optimizer that minimizes a loss function.
pub trait Optimizer: BaseOptimizer {
    /// Perform a loss minimization step using the gradient of a loss function.
    ///
    /// Obtains gradients by backpropagating the result of `loss_fn`.
    ///
    /// # Args
    /// * `loss_fn` - Loss function to minimize.
    ///     Called to obtain the loss tensor, which is back-propagated to obtain a gradient.
    ///     Always evaluated at least once; may be evaluated multiple times.
    /// * `logger` - Logger for statistics and other information about the step.
    ///
    /// # Returns
    /// The initial value of `loss_fn` on success.
    ///
    /// If an error is detected, the parameters are guaranteed to be unchanged from (or reset to)
    /// their initial values.
    /// In general, error conditions are not guaranteed to be detected and an optimizer
    /// may silently put itself or the parameters into a bad state.
    /// For example, [`COptimizer`] sets parameters to NaN when the loss is NaN.
    ///
    /// [`COptimizer`]: tch::COptimizer
    fn backward_step(
        &mut self,
        loss_fn: &dyn Fn() -> Tensor,
        logger: &mut dyn Logger,
    ) -> Result<Tensor, OptimizerStepError>;
}

/// Optimizer that minimizes a loss tensor using a single gradient evaluation per step.
///
/// Specifically, each step may use the gradient at the initial point of the step
/// and makes no further gradient evaluations.
pub trait OnceOptimizer: BaseOptimizer {
    /// Perform a loss minimization step (parameter update).
    ///
    /// Uses the existing gradients stored with the parameter tensor.
    ///
    /// If an error is detected, the parameters are guaranteed to be unchanged from (or reset to)
    /// their initial values.
    /// In general, error conditions are not guaranteed to be detected and an optimizer
    /// may silently put itself or the parameters into a bad state.
    /// For example, [`COptimizer`] sets parameters to NaN when the loss is NaN.
    ///
    /// [`COptimizer`]: tch::COptimizer
    fn step_once(&self, logger: &mut dyn Logger) -> Result<(), OptimizerStepError>;

    /// Apply a backward step pass, update the gradients, and perform an optimization step.
    ///
    /// Obtains gradients by backpropagating the result of `loss_fn`.
    ///
    /// # Args
    /// * `loss` - Loss tensor. Back-propagation is applied to this tensor to obtain a gradient.
    /// * `logger` - Logger for statistics and other information about the step.
    ///
    /// # Returns
    /// The initial value of `loss_fn` on success.
    ///
    /// If an error is detected, the parameters are guaranteed to be unchanged from (or reset to)
    /// their initial values.
    /// In general, error conditions are not guaranteed to be detected and an optimizer
    /// may silently put itself or the parameters into a bad state.
    /// For example, [`COptimizer`] sets parameters to NaN when the loss is NaN.
    ///
    /// [`COptimizer`]: tch::COptimizer
    fn backward_step_once(
        &mut self,
        loss: &Tensor,
        logger: &mut dyn Logger,
    ) -> Result<(), OptimizerStepError>;
}

impl<T: OnceOptimizer> Optimizer for T {
    fn backward_step(
        &mut self,
        loss_fn: &dyn Fn() -> Tensor,
        logger: &mut dyn Logger,
    ) -> Result<Tensor, OptimizerStepError> {
        let loss = loss_fn();
        self.backward_step_once(&loss, logger)?;
        Ok(loss)
    }
}

/// Optimizer that minimizes a loss function subject to a trust region constraint on each step.
pub trait TrustRegionOptimizer: BaseOptimizer {
    /// Take an optimization step subject to a distance constraint
    ///
    /// This function obtains gradients by backpropagating the result of `loss_distance_fn`,
    /// once for each `loss` and `distance`.
    /// It is not necessary for the caller to compute or zero out the existing gradients.
    ///
    /// # Args
    /// * `loss_distance_fn` - Function returning scalar loss and distance values.
    ///     * Loss is minimized.
    ///     * The non-negative distance value measures a deviation of the current parameter values
    ///         from the initial parameters at the start of this step.
    ///         It should equal zero at the start of the step.
    ///
    /// * `max_distance` - Upper bound on the distance value for this step.
    ///
    /// * `logger` - Logger for statistics and other information about the step.
    ///
    /// # Returns
    /// The initial loss value on success.
    fn trust_region_backward_step(
        &self,
        loss_distance_fn: &dyn Fn() -> (Tensor, Tensor),
        max_distance: f64,
        logger: &mut dyn Logger,
    ) -> Result<f64, OptimizerStepError>;
}

/// Error performing an optimization step.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum OptimizerStepError {
    #[error("loss is not improving: (new) {loss} >= (prev) {loss_before}")]
    LossNotImproving { loss: f64, loss_before: f64 },
    #[error(
        "constraint is violated: (val) {constraint_val} >= (threshold) {max_constraint_value}"
    )]
    ConstraintViolated {
        constraint_val: f64,
        max_constraint_value: f64,
    },
    #[error("loss is NaN")]
    NaNLoss,
    #[error("constraint is NaN")]
    NaNConstraint,
}

/// Build an optimizer
pub trait BuildOptimizer {
    type Optimizer;
    type Error: Error;

    /// Build an optimizer for the trainable variables in a variable store.
    fn build_optimizer(&self, vs: &VarStore) -> Result<Self::Optimizer, Self::Error>;
}

#[cfg(test)]
mod testing {
    use super::*;
    use tch::{Device, Kind};

    pub fn check_optimizes_quadratic<OC>(optimizer_config: &OC, num_steps: u64)
    where
        OC: BuildOptimizer,
        OC::Optimizer: Optimizer,
    {
        // Minimize f(x) = 1/2*x'Mx + b'x
        // with M = [1  -1]  b = [ 2]
        //          [-1  2]      [-3]
        //
        // which is minimized at x = [-1  1]'
        let m = Tensor::of_slice(&[1.0_f32, -1.0, -1.0, 2.0]).reshape(&[2, 2]);
        let b = Tensor::of_slice(&[2.0_f32, -3.0]);

        let vs = VarStore::new(Device::Cpu);
        let x = vs.root().f_zeros("x", &[2]).unwrap();
        let mut optimizer = optimizer_config.build_optimizer(&vs).unwrap();

        let loss_fn = || m.mv(&x).dot(&x) / 2 + b.dot(&x);

        for _ in 0..num_steps {
            let _ = optimizer.backward_step(&loss_fn, &mut ()).unwrap();
        }

        let expected = Tensor::of_slice(&[-1.0, 1.0]);
        assert!(
            f64::from((&x - &expected).norm()) < 1e-3,
            "expected: {:?}, actual: {:?}",
            expected,
            x
        );
    }

    pub fn check_trust_region_optimizes_quadratic<OC>(optimizer_config: &OC, num_steps: u64)
    where
        OC: BuildOptimizer,
        OC::Optimizer: TrustRegionOptimizer,
    {
        // Minimize f(x) = 1/2*x'Mx + b'x
        // with M = [1  -1]  b = [ 2]
        //          [-1  2]      [-3]
        //
        // which is minimized at x = [-1  1]'
        let m = Tensor::of_slice(&[1.0_f32, -1.0, -1.0, 2.0]).reshape(&[2, 2]);
        let b = Tensor::of_slice(&[2.0_f32, -3.0]);

        let vs = VarStore::new(Device::Cpu);
        let x = vs.root().zeros("x", &[2]);
        let optimizer = optimizer_config.build_optimizer(&vs).unwrap();

        let x_last = x.detach().copy();
        let loss_distance_fn = || {
            let loss = m.mv(&x).dot(&x) / 2 + b.dot(&x);
            let distance = (&x - &x_last).square().sum(Kind::Float);
            (loss, distance)
        };

        for _ in 0..num_steps {
            x_last.detach().copy_(&x);
            let result = optimizer.trust_region_backward_step(&loss_distance_fn, 0.001, &mut ());
            match result {
                Err(OptimizerStepError::LossNotImproving {
                    loss: _,
                    loss_before: _,
                }) => break,
                r => r.unwrap(),
            };
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
