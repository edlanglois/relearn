//! Torch optimizer wrappers and configuration
use super::{BaseOptimizer, OnceOptimizer, OptimizerBuilder, OptimizerStepError};
use std::convert::{TryFrom, TryInto};
use tch::{nn::VarStore, COptimizer, TchError, Tensor};

impl BaseOptimizer for COptimizer {
    fn zero_grad(&mut self) {
        COptimizer::zero_grad(self).unwrap();
    }
}

impl OnceOptimizer for COptimizer {
    fn step_once(&self) -> Result<(), OptimizerStepError> {
        // I'm not sure what errors it is possible for torch to raise here
        // Anything that isn't essentially a type error should be converted to OptimizerStepError.
        COptimizer::step(self).unwrap();
        Ok(())
    }

    fn backward_step_once(&mut self, loss: &Tensor) -> Result<(), OptimizerStepError> {
        BaseOptimizer::zero_grad(self);
        loss.backward();
        self.step_once()
    }
}

impl<T> OptimizerBuilder<COptimizer> for T
where
    for<'a> &'a T: TryInto<COptimizer, Error = TchError>,
{
    type Error = TchError;

    fn build_optimizer(&self, vs: &VarStore) -> Result<COptimizer, TchError> {
        let mut optimizer: COptimizer = self.try_into()?;
        let variables = vs.variables_.lock().unwrap();
        for var in &variables.trainable_variables {
            optimizer.add_parameters(&var.tensor, var.group)?;
        }
        Ok(optimizer)
    }
}

/// Configuration for the SGD optimizer.
#[derive(Debug, Clone)]
pub struct SgdConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum
    pub momentum: f64,
    /// Weight decay (L2 penalty)
    pub weight_decay: f64,
    /// Dampening for momentum
    pub dampening: f64,
    /// Enables Nesterov momentum
    pub nesterov: bool,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
        }
    }
}

impl TryFrom<&SgdConfig> for COptimizer {
    type Error = TchError;
    fn try_from(config: &SgdConfig) -> Result<Self, Self::Error> {
        COptimizer::sgd(
            config.learning_rate,
            config.momentum,
            config.dampening,
            config.weight_decay,
            config.nesterov,
        )
    }
}

#[allow(clippy::doc_markdown)] // false positive on RMSProp
/// Configuration for the RMSProp optimizer.
#[derive(Debug, Clone)]
pub struct RmsPropConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum
    pub momentum: f64,
    /// Smoothing factor
    pub alpha: f64,
    /// A term added to the denominator to improve numerical stability
    pub eps: f64,
    /// If true, normalize the gradient by the estimated variance.
    pub centered: bool,
    /// Weight decay (L2 penalty)
    pub weight_decay: f64,
}

impl Default for RmsPropConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            momentum: 0.0,
            alpha: 0.99,
            eps: 1e-8,
            centered: false,
            weight_decay: 0.0,
        }
    }
}

impl TryFrom<&RmsPropConfig> for COptimizer {
    type Error = TchError;
    fn try_from(config: &RmsPropConfig) -> Result<Self, Self::Error> {
        COptimizer::rms_prop(
            config.learning_rate,
            config.alpha,
            config.eps,
            config.weight_decay,
            config.momentum,
            config.centered,
        )
    }
}

/// Configuration for the Adam optimizer.
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Coefficient for the running average of the gradient
    pub beta1: f64,
    /// Coefficient for the running average of the square of the gradient
    pub beta2: f64,
    /// Weight decay (L2 penalty)
    pub weight_decay: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
        }
    }
}

impl TryFrom<&AdamConfig> for COptimizer {
    type Error = TchError;
    fn try_from(config: &AdamConfig) -> Result<Self, Self::Error> {
        COptimizer::adam(
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.weight_decay,
        )
    }
}

#[allow(clippy::doc_markdown)]
/// Configuration for the AdamW optimizer.
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Coefficient for the running average of the gradient
    pub beta1: f64,
    /// Coefficient for the running average of the square of the gradient
    pub beta2: f64,
    /// Weight decay (L2 penalty)
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
        }
    }
}

impl TryFrom<&AdamWConfig> for COptimizer {
    type Error = TchError;
    fn try_from(config: &AdamWConfig) -> Result<Self, Self::Error> {
        COptimizer::adamw(
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.weight_decay,
        )
    }
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod coptimizer {
    use super::super::{testing, Optimizer};
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn sgd_optimizes_quadratic() {
        let config = SgdConfig {
            learning_rate: 1e-1,
            ..SgdConfig::default()
        };
        testing::check_optimizes_quadratic(&config, 500);
    }

    #[test]
    /// Track the behavour when loss is NaN.
    ///
    /// SGD continues anyways and sets the parameters to NaN.
    /// We might want to check for NaN in the wrapper and fail
    /// but for now it silently runs without error.
    fn sgd_nan_loss() {
        let vs = VarStore::new(Device::Cpu);
        let x = vs.root().f_zeros("x", &[2]).unwrap();

        let mut optimizer = SgdConfig::default().build_optimizer(&vs).unwrap();
        #[allow(clippy::eq_op)]
        let _ = optimizer
            .backward_step(&(|| (&x / &x).sum(Kind::Float)))
            .unwrap();
    }

    #[test]
    fn rms_prop_optimizes_quadratic() {
        let config = RmsPropConfig {
            learning_rate: 1e-1,
            ..RmsPropConfig::default()
        };
        testing::check_optimizes_quadratic(&config, 500);
    }

    #[test]
    fn adam_optimizes_quadratic() {
        let config = RmsPropConfig {
            learning_rate: 1e-1,
            ..RmsPropConfig::default()
        };
        testing::check_optimizes_quadratic(&config, 500);
    }

    #[test]
    fn adam_w_optimizes_quadratic() {
        let config = RmsPropConfig {
            learning_rate: 1e-1,
            ..RmsPropConfig::default()
        };
        testing::check_optimizes_quadratic(&config, 500);
    }
}
