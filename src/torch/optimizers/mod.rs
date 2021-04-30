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
    fn f_step(&self) -> Result<(), Self::Error>;

    /// Perform a single optimization step (parameter update).
    fn step(&self) {
        self.f_step().unwrap();
    }

    /// Applies a backward step pass, updates the gradients, and performs an optimization step.
    fn f_backward_step(&self, loss: &Tensor) -> Result<(), Self::Error>;

    /// Applies a backward step pass, updates the gradients, and performs an optimization step.
    fn backward_step(&self, loss: &Tensor) {
        self.f_backward_step(loss).unwrap();
    }
}

/// Build an optimizer
pub trait OptimizerBuilder<T> {
    type Error: Error;

    /// Build an optimizer for the trainable variables in a variable store.
    fn build_optimizer(&self, vs: &VarStore) -> Result<T, Self::Error>;
}
