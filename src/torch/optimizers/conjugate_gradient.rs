//! Conjugate gradient optimizer
//!
//! Based on the [Python CGD implementation][garage_cgo] of the
//! [Garage Toolkit](https://github.com/rlworkgroup/garage).
//!
//! [garage_cgo]: https://github.com/rlworkgroup/garage/blob/90b60905b29cea8f8373c6732ced0cadf8489b0c/src/garage/torch/optimizers/conjugate_gradient_optimizer.py

// == MIT License For This File Only ==
//
// Copyright (c) 2019 Reinforcement Learning Working Group
// Copyright (c) 2021 Eric Langlois
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use super::super::utils;
use super::{BaseOptimizer, OptimizerBuilder, OptimizerStepError, TrustRegionOptimizer};
use std::borrow::Borrow;
use std::convert::Infallible;
use tch::{nn::VarStore, Tensor};
use thiserror::Error;

/// Configuration for the Conjugate Gradient Optimizer
#[derive(Debug, Clone)]
pub struct ConjugateGradientOptimizerConfig {
    /// Number of CG iterations used to calculate A^-1 g
    pub cg_iters: u64,
    /// Maximum number of iterations for backtracking line search
    pub max_backtracks: u64,
    /// Backtrack ratio for backtracking line search
    pub backtrack_ratio: f64,
    /// A small value so that A -> A + reg*I. It is used by Hessian Vector Product calculation.
    pub hpv_reg_coeff: f64,
    /// Whether to accept the descent step if it violates the line search condition
    /// after exhausing all backtracking budgets.
    pub accept_violation: bool,
}

impl Default for ConjugateGradientOptimizerConfig {
    fn default() -> Self {
        Self {
            cg_iters: 10,
            max_backtracks: 15,
            backtrack_ratio: 0.8,
            hpv_reg_coeff: 1e-5,
            accept_violation: false,
        }
    }
}

impl OptimizerBuilder<ConjugateGradientOptimizer> for ConjugateGradientOptimizerConfig {
    type Error = Infallible;

    fn build_optimizer(&self, vs: &VarStore) -> Result<ConjugateGradientOptimizer, Infallible> {
        Ok(ConjugateGradientOptimizer::new(vs, self.clone()))
    }
}

/// Conjugate gradient optimizer
///
/// Performs constrained optimization via backtracking line search.
///
/// The search direction is computed using a conjugate gradient algorithm,
/// which gives `x = A^{-1}g`, where `A` is a second order approximation of
/// the constraint and `g` is the gradient of the loss function.
#[derive(Debug)]
pub struct ConjugateGradientOptimizer {
    /// Parameters to optimize
    params: Vec<Tensor>,
    config: ConjugateGradientOptimizerConfig,
}

impl ConjugateGradientOptimizer {
    pub fn new(vs: &VarStore, config: ConjugateGradientOptimizerConfig) -> Self {
        Self {
            params: vs.trainable_variables(),
            config,
        }
    }
}

impl BaseOptimizer for ConjugateGradientOptimizer {
    fn zero_grad(&self) {
        for param in self.params.iter() {
            utils::zero_grad(param);
        }
    }
}

impl TrustRegionOptimizer for ConjugateGradientOptimizer {
    fn trust_region_backward_step(
        &self,
        loss_fn: &dyn Fn() -> Tensor,
        distance_fn: &dyn Fn() -> Tensor,
        max_distance: f64,
    ) -> Result<Tensor, OptimizerStepError> {
        let loss = loss_fn();
        self.zero_grad();
        loss.backward();
        self.trust_region_step(loss_fn, distance_fn, max_distance);
        Ok(loss)
    }
}

impl ConjugateGradientOptimizer {
    /// Take an optimization step subject to a constraint function.
    ///
    /// # Args
    /// * `loss_fn` - Forward loss function. Used to evaluate loss at the current parameters.
    ///               Not used to evaluate the gradient: the existing stored gradient is used.
    ///
    /// * `distance_fn` - Function that measures the distance of the current parameter values from
    ///     the initial parameters.
    ///
    /// * `max_distance` - Upper bound on `distance_fn` for this step.
    pub fn trust_region_step<F, G>(&self, loss_fn: &F, distance_fn: &G, max_distance: f64)
    where
        F: Fn() -> Tensor + ?Sized,
        G: Fn() -> Tensor + ?Sized,
    {
        let flat_loss_grads = utils::flatten_tensors(self.params.iter().map(Tensor::grad));

        // Build Hessian-vector-product function
        let hvp_fn =
            HessianVectorProduct::new(distance_fn, &self.params, self.config.hpv_reg_coeff);

        // Compute step direction
        let mut step_dir =
            conjugate_gradient(&hvp_fn, &flat_loss_grads, self.config.cg_iters, 1e-10);
        // Replace nan with 0 (also +- inf with largest/smallest values)
        let _ = step_dir.nan_to_num_(0.0, None, None);

        // Compute step size
        let step_size = match ((f64::from(step_dir.dot(&hvp_fn.eval(&step_dir))) + 1e-8).recip()
            * max_distance
            * 2.0)
            .sqrt()
        {
            x if x.is_nan() => 1.0,
            x => x,
        };

        let descent_step = step_size * step_dir;
        // These errors all relate to a point on the line search, not necessarily the original
        // paramter values, which might be fine.
        let _ = self.backtracking_line_search(&descent_step, loss_fn, distance_fn, max_distance);
    }

    fn backtracking_line_search<F, G>(
        &self,
        descent_step: &Tensor,
        loss_fn: &F,
        constraint_fn: &G,
        max_constraint_value: f64,
    ) -> Result<(), LineSearchError>
    where
        F: Fn() -> Tensor + ?Sized,
        G: Fn() -> Tensor + ?Sized,
    {
        let mut params: Vec<_> = self.params.iter().map(Tensor::detach).collect();
        let prev_params: Vec<_> = params.iter().map(Tensor::copy).collect();
        let param_shapes: Vec<_> = self.params.iter().map(Tensor::size).collect();

        let descent_step = utils::unflatten_tensors(descent_step, &param_shapes);
        let loss_before: f64 = loss_fn().into();

        let mut loss = loss_before;
        let mut constraint_val = f64::INFINITY;
        for i in 0..self.config.max_backtracks {
            let ratio = self.config.backtrack_ratio.powi(i as i32);

            for ((step, prev_param), param) in descent_step
                .iter()
                .zip(prev_params.iter())
                .zip(params.iter_mut())
            {
                // Garage uses param.data() but its usage is deprecated
                // https://pytorch.org/blog/pytorch-0_4_0-migration-guide/#what-about-data
                // Using detach instead (in definition of params)
                param.copy_(&(prev_param - ratio * step));
            }

            loss = loss_fn().into();
            constraint_val = constraint_fn().into();
            if loss < loss_before && constraint_val <= max_constraint_value {
                break;
            }
        }

        let result = if loss.is_nan() {
            Err(LineSearchError::NaNLoss)
        } else if constraint_val.is_nan() {
            Err(LineSearchError::NaNConstraint)
        } else if loss >= loss_before {
            Err(LineSearchError::LossNotImproving { loss, loss_before })
        } else if constraint_val >= max_constraint_value && !self.config.accept_violation {
            Err(LineSearchError::ConstraintViolated {
                constraint_val,
                max_constraint_value,
            })
        } else {
            Ok(())
        };

        if result.is_err() {
            // Reset the parameter values
            for (param, prev_param) in params.iter_mut().zip(prev_params) {
                param.copy_(&prev_param);
            }
        }

        result
    }
}

#[derive(Error, Debug)]
pub enum LineSearchError {
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

/// Implements a Hessian-vector product function
///
/// # Reference
/// Pearlmutter, Barak A. "Fast exact multiplication by the Hessian."
/// Neural computation 6.1 (1994): 147-160.`
struct HessianVectorProduct<'a, T> {
    /// Parameter tensors
    params: &'a [T],
    /// Regularization coefficient.
    reg_coeff: f64,
    /// The shape of each tensor in params.
    param_shapes: Vec<Vec<i64>>,
    /// Gradients with respect to each of `params`.
    f_grads: Vec<Tensor>,
}

impl<'a, T> HessianVectorProduct<'a, T>
where
    T: Borrow<Tensor>,
{
    /// Create a new Hessian-vector product function
    ///
    /// # Args
    /// * `f` - Use the Hessian of this function.
    /// * `params` - A list of function parameter tensors.
    /// * `reg_coeff` - Regularization coefficient. A small value so that A -> A + reg*I.
    pub fn new<F>(f: &F, params: &'a [T], reg_coeff: f64) -> Self
    where
        F: Fn() -> Tensor + ?Sized,
    {
        let param_shapes = params.iter().map(|t| t.borrow().size()).collect();
        let f_out = f();
        let f_grads = Tensor::run_backward(&[f_out], params, true, true);
        Self {
            params,
            reg_coeff,
            param_shapes,
            f_grads,
        }
    }

    /// Evaluate the product of this Hessian with an appropriately sized vector.
    pub fn eval(&self, vector: &Tensor) -> Tensor {
        let unflattened_vector = utils::unflatten_tensors(vector, &self.param_shapes);

        assert_eq!(self.f_grads.len(), unflattened_vector.len());
        let grad_vector_product = Tensor::stack(
            &self
                .f_grads
                .iter()
                .zip(&unflattened_vector)
                .map(|(g, x)| utils::flat_dot(g, x))
                .collect::<Vec<_>>(),
            0,
        )
        .sum(vector.kind());

        let hpv = Tensor::run_backward(&[grad_vector_product], self.params, true, false);

        // Note:
        // The garage implementation here checks if any hpv[i] is None and, if so,
        // sets hpv[i] = zeros_like(params[i]).
        // The run_backward signature is Vec<Tensor> not Vec<Option<Tensor>> so
        // I assume that missing gradients are not possible with this API.

        let flat_output = utils::flatten_tensors(&hpv);
        // flat_output + reg_coeff * vector
        flat_output.g_add(&vector.g_mul1(self.reg_coeff))
    }
}

/// Use Conjugate Gradient iteration to solve Ax = b. Demmel p 312.
///
/// # Args
/// * `f_Ax` - Computes the Hessian-vector product.
/// * `b` - Right hand side of the equation to solve.
/// * `cg_iters` - Number of iterations to run the conjugate gradient algorithm.
/// * `residual_tol`: Tolerance for convergence.
///
/// # Returns
/// Solution x* for equation Ax = b.
fn conjugate_gradient<T: Borrow<Tensor>>(
    hvp_fn: &HessianVectorProduct<T>,
    b: &Tensor,
    cg_iters: u64,
    residual_tol: f64,
) -> Tensor {
    let mut p = b.copy();
    let mut r = b.copy();
    let mut x = b.zeros_like();
    let mut r_dot_r = r.dot(&r);

    for _ in 0..cg_iters {
        let z = hvp_fn.eval(&p);
        let v = &r_dot_r / p.dot(&z);
        let _ = x.addcmul_(&v, &p);
        r -= (&v) * (&z);
        let new_r_dot_r = r.dot(&r);
        let mu = &new_r_dot_r / &r_dot_r;
        // Note: Garage version does not re-use the memory of p despite the initial clone()
        let _ = p.g_mul_(&mu);
        let _ = p.g_add_(&r);

        r_dot_r = new_r_dot_r;
        if f64::from(&r_dot_r) < residual_tol {
            break;
        }
    }
    x
}

#[cfg(test)]
mod conjugate_gradient {
    use super::super::testing;
    use super::*;

    #[test]
    fn optimizes_quadratic() {
        let config = ConjugateGradientOptimizerConfig::default();
        testing::check_trust_region_optimizes_quadratic(&config, 500);
    }
}

#[cfg(test)]
mod hessian_vector_product {
    use super::*;
    use tch::{Cuda, Device, Kind};

    #[test]
    fn quadratic_hessian() {
        // Work-around for https://github.com/pytorch/pytorch/issues/35736
        Cuda::is_available();

        // f(x) = 1/2*x'Mx + b'x
        // H_f(x) = M
        let m = Tensor::of_slice(&[1.0f32, -1.0, -1.0, 2.0]).reshape(&[2, 2]);
        let b = Tensor::of_slice(&[2.0f32, -3.0]);
        let x = Tensor::zeros(&[2], (Kind::Float, Device::Cpu)).requires_grad_(true);
        let f = || m.mv(&x).dot(&x) / 2 + b.dot(&x);
        let params = [&x];

        let hvp = HessianVectorProduct::new(&f, &params, 0.0);

        assert_eq!(
            hvp.eval(&Tensor::of_slice(&[1.0f32, 0.0])),
            Tensor::of_slice(&[1.0f32, -1.0])
        );
        assert_eq!(
            hvp.eval(&Tensor::of_slice(&[0.0f32, 1.0])),
            Tensor::of_slice(&[-1.0f32, 2.0])
        );
    }
}
