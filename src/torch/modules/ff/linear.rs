//! Linear layer
use super::super::{BuildModule, FeedForwardModule, Module};
use crate::torch::initializers::Initializer;
use tch::{nn::Path, Tensor};

/// Configuration for the [`Linear`] module.
#[derive(Debug, Copy, Clone)]
pub struct LinearConfig {
    /// Initializer for the kernel (weight) matrix.
    kernel_init: Initializer,
    /// Initializer for the bias vector, if one exists.
    bias_init: Option<Initializer>,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            // TODO: Consider switching to Orthogonal
            // https://arxiv.org/pdf/2001.05992.pdf
            kernel_init: Initializer::default(),
            bias_init: Some(Initializer::default()),
        }
    }
}

impl BuildModule for LinearConfig {
    type Module = Linear;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        Linear::new(vs, in_dim, out_dim, self)
    }
}

/// Linear fully-connected layer module.
#[derive(Debug)]
pub struct Linear {
    kernel: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(vs: &Path, in_dim: usize, out_dim: usize, config: &LinearConfig) -> Self {
        // Total fan_in is the weigths in_dim + 1 for the bias.
        let fan_in = (in_dim + 1) as i64;
        Self {
            kernel: config.kernel_init.add_tensor(
                vs,
                "kernel",
                &[out_dim as i64, in_dim as i64],
                1.0,
                Some(fan_in),
            ),
            bias: config
                .bias_init
                .map(|init| init.add_tensor(vs, "bias", &[out_dim as i64], 1.0, Some(fan_in))),
        }
    }
}

impl Module for Linear {}

impl FeedForwardModule for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.linear(&self.kernel, self.bias.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{testing, AsSeq};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::nn::VarStore;
    use tch::{kind::Kind, Device};

    #[fixture]
    fn default_module() -> (Linear, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = LinearConfig::default();
        let vs = VarStore::new(Device::Cpu);
        let module = config.build_module(&vs.root(), in_dim, out_dim);
        (module, in_dim, out_dim)
    }

    #[rstest]
    fn default_forward_batch(default_module: (Linear, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_forward(&default_mlp, in_dim, out_dim, &[4], Kind::Float);
    }

    #[rstest]
    fn default_seq_serial(default_module: (Linear, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_seq_serial(&AsSeq::new(default_mlp), in_dim, out_dim);
    }

    #[rstest]
    fn default_step(default_module: (Linear, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_step(&AsSeq::new(default_mlp), in_dim, out_dim);
    }

    #[test]
    fn default_forward_gradient_descent() {
        testing::check_config_forward_gradient_descent(&LinearConfig::default());
    }
}
