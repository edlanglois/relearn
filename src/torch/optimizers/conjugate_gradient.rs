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
        loss_distance_fn: &dyn Fn() -> (Tensor, Tensor),
        max_distance: f64,
    ) -> Result<f64, OptimizerStepError> {
        let (loss, distance) = loss_distance_fn();

        // Loss gradient. Save the graph so that HessianVectorProduct can reuse it.
        self.zero_grad();
        let loss_grads = Tensor::run_backward(&[&loss], &self.params, true, false);
        let flat_loss_grads = utils::flatten_tensors(loss_grads);

        // Build Hessian-vector-product function. Backpropagates distance gradients.
        let hvp_fn = HessianVectorProduct::new(&distance, &self.params, self.config.hpv_reg_coeff);

        // Compute step direction
        let mut step_dir =
            conjugate_gradient(&hvp_fn, &flat_loss_grads, self.config.cg_iters, 1e-10);
        // Replace nan with 0 (also +- inf with largest/smallest values)
        let _ = step_dir.nan_to_num_(0.0, None, None);

        // Compute step size
        let step_size = match ((f64::from(step_dir.dot(&hvp_fn.mat_vec_mul(&step_dir))) + 1e-8)
            .recip()
            * max_distance
            * 2.0)
            .sqrt()
        {
            x if x.is_nan() => 1.0,
            x => x,
        };

        let descent_step = step_size * step_dir;
        let initial_loss: f64 = loss.into();

        self.backtracking_line_search(&descent_step, loss_distance_fn, max_distance, initial_loss)?;

        Ok(initial_loss)
    }
}

impl ConjugateGradientOptimizer {
    fn backtracking_line_search<F>(
        &self,
        descent_step: &Tensor,
        loss_constraint_fn: &F,
        max_constraint_value: f64,
        initial_loss: f64,
    ) -> Result<(), OptimizerStepError>
    where
        F: Fn() -> (Tensor, Tensor) + ?Sized,
    {
        let mut params: Vec<_> = self.params.iter().map(Tensor::detach).collect();
        let prev_params: Vec<_> = params.iter().map(Tensor::copy).collect();
        let param_shapes: Vec<_> = self.params.iter().map(Tensor::size).collect();

        let descent_step = utils::unflatten_tensors(descent_step, &param_shapes);

        let mut loss = initial_loss;
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

            let (loss_tensor, constraint_tensor) = loss_constraint_fn();
            loss = loss_tensor.into();
            constraint_val = constraint_tensor.into();
            if loss < initial_loss && constraint_val <= max_constraint_value {
                break;
            }
        }

        let result = if loss.is_nan() {
            Err(OptimizerStepError::NaNLoss)
        } else if constraint_val.is_nan() {
            Err(OptimizerStepError::NaNConstraint)
        } else if loss >= initial_loss {
            Err(OptimizerStepError::LossNotImproving {
                loss,
                loss_before: initial_loss,
            })
        } else if constraint_val >= max_constraint_value && !self.config.accept_violation {
            Err(OptimizerStepError::ConstraintViolated {
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
    grads: Vec<Tensor>,
}

impl<'a, T> HessianVectorProduct<'a, T>
where
    T: Borrow<Tensor>,
{
    /// Create a new Hessian-vector product function
    ///
    /// Evaluates the Hessian of the mapping `params -> output`.
    /// Zeros the existing gradients and backpropagates gradients from `output`.
    ///
    /// # Args
    /// * `output` - Function output tensor.
    /// * `params` - A list of function parameter tensors.
    /// * `reg_coeff` - Regularization coefficient. A small value so that A -> A + reg*I.
    pub fn new(output: &Tensor, params: &'a [T], reg_coeff: f64) -> Self {
        let param_shapes = params.iter().map(|t| t.borrow().size()).collect();
        for param in params.iter() {
            utils::zero_grad(param.borrow());
        }
        // TODO: Possible memory leak. Unset param.grad afterward?
        // https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward
        let grads = Tensor::run_backward(&[output], params, true, true);
        Self {
            params,
            reg_coeff,
            param_shapes,
            grads,
        }
    }
}

impl<'a, T> MatrixVectorProduct for HessianVectorProduct<'a, T>
where
    T: Borrow<Tensor>,
{
    type Vector = Tensor;

    fn mat_vec_mul(&self, vector: &Tensor) -> Tensor {
        let unflattened_vector = utils::unflatten_tensors(vector, &self.param_shapes);

        assert_eq!(self.grads.len(), unflattened_vector.len());
        let grad_vector_product = Tensor::stack(
            &self
                .grads
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

/// A Matrix-Vector product
pub trait MatrixVectorProduct {
    type Vector;

    fn mat_vec_mul(&self, vector: &Self::Vector) -> Self::Vector;
}

impl MatrixVectorProduct for Tensor {
    type Vector = Tensor;

    fn mat_vec_mul(&self, vector: &Self::Vector) -> Self::Vector {
        self.mv(vector)
    }
}

/// Use Conjugate Gradient iteration to solve `Ax = b` where `A` is symmetric positive definite.
///
/// # Args
/// * `f_Ax` - Computes the Hessian-vector product.
/// * `b` - Right hand side of the equation to solve.
/// * `cg_iters` - Number of iterations to run the conjugate gradient algorithm.
/// * `residual_tol`: Tolerance for convergence.
///
/// # Returns
/// Solution `x*` for the equation `Ax = b`.
///
/// # Reference
/// https://en.wikipedia.org/wiki/Conjugate_gradient_method
fn conjugate_gradient<T: MatrixVectorProduct<Vector = Tensor>>(
    #[allow(non_snake_case)] f_Ax: &T,
    b: &Tensor,
    cg_iters: u64,
    residual_tol: f64,
) -> Tensor {
    let mut x = b.zeros_like();
    let mut residual = b.copy(); // b - Ax where x = 0

    // step direction (p). residual projected to be orthogonal to previous steps
    let mut step = b.copy();
    let mut residual_norm_squared = residual.dot(&residual);

    for _ in 0..cg_iters {
        let z = f_Ax.mat_vec_mul(&step); // A *  step
        let alpha = &residual_norm_squared / step.dot(&z); // ||r||^2 / (step' * A * step)
        let _ = x.addcmul_(&alpha, &step); // x += alpha * step
        let _ = residual.addcmul_(&(-alpha), &z); // r -= alpha * A*step

        let new_residual_norm_squared = residual.dot(&residual);
        if f64::from(&new_residual_norm_squared) < residual_tol {
            break;
        }

        let mu = &new_residual_norm_squared / &residual_norm_squared;
        // Note: Garage version does not re-use the memory of p despite the initial clone()
        let _ = step.g_mul_(&mu);
        let _ = step.g_add_(&residual);

        residual_norm_squared = new_residual_norm_squared;
    }
    x
}

#[cfg(test)]
mod cg_optimizer {
    use super::super::testing;
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn optimizes_quadratic() {
        let config = ConjugateGradientOptimizerConfig::default();
        testing::check_trust_region_optimizes_quadratic(&config, 500);
    }

    #[test]
    fn shared_loss_distance_computation() {
        let config = ConjugateGradientOptimizerConfig::default();

        let vs = VarStore::new(Device::Cpu);
        let x = vs.root().ones("x", &[2]);
        let optimizer = config.build_optimizer(&vs).unwrap();

        let y_prev = x.square().mean(Kind::Float).detach();
        let loss_distance_fn = || {
            let y = x.square().mean(Kind::Float);
            let loss = &y + 1.0;
            let distance = (&y - &y_prev).square();
            (loss, distance)
        };

        for _ in 0..100 {
            y_prev
                .detach()
                .copy_(&x.square().mean(Kind::Float).detach());
            let result = optimizer.trust_region_backward_step(&loss_distance_fn, 0.001);
            match result {
                Err(OptimizerStepError::LossNotImproving {
                    loss: _,
                    loss_before: _,
                }) => break,
                r => r.unwrap(),
            };
        }

        let expected = Tensor::of_slice(&[0.0, 0.0]);
        assert!(
            f64::from((&x - &expected).norm()) < 0.1,
            "expected: {:?}, actual: {:?}",
            expected,
            x
        );
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
        let y = m.mv(&x).dot(&x) / 2 + b.dot(&x);
        let params = [&x];

        let hvp = HessianVectorProduct::new(&y, &params, 0.0);

        assert_eq!(
            hvp.mat_vec_mul(&Tensor::of_slice(&[1.0f32, 0.0])),
            Tensor::of_slice(&[1.0f32, -1.0])
        );
        assert_eq!(
            hvp.mat_vec_mul(&Tensor::of_slice(&[0.0f32, 1.0])),
            Tensor::of_slice(&[-1.0f32, 2.0])
        );
    }
}
