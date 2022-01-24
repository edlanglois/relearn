//! Multi-layer perceptron
use super::super::{BuildModule, FeedForwardModule, Module};
use super::func::Activation;
use std::iter;
use tch::{
    nn::{self, Linear, LinearConfig, Path},
    Tensor,
};

/// Configuration for the [`Mlp`] module.
#[derive(Debug, Clone)]
pub struct MlpConfig {
    /// Sizes of the hidden layers
    pub hidden_sizes: Vec<usize>,
    /// Activation function between hidden layers.
    pub activation: Activation,
    /// Activation function on the output.
    pub output_activation: Activation,
    /// Configuration for the linear layers
    pub linear_config: LinearConfig,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            hidden_sizes: vec![128],
            activation: Activation::Relu,
            output_activation: Activation::Identity,
            linear_config: LinearConfig::default(),
        }
    }
}

impl BuildModule for MlpConfig {
    type Module = Mlp;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        Mlp::new(vs, in_dim, out_dim, self)
    }
}

/// Multi-layer perceptron
pub struct Mlp {
    layers: Vec<Linear>,
    activation: Option<fn(&Tensor) -> Tensor>,
    output_activation: Option<fn(&Tensor) -> Tensor>,
}

impl Mlp {
    pub fn new(vs: &Path, in_dim: usize, out_dim: usize, config: &MlpConfig) -> Self {
        let in_dims = iter::once(&in_dim).chain(&config.hidden_sizes);
        let out_dims = config.hidden_sizes.iter().chain(iter::once(&out_dim));

        let layers: Vec<_> = in_dims
            .zip(out_dims)
            .enumerate()
            .map(|(i, (in_, out_))| {
                nn::linear(
                    vs / format!("layer_{}", i),
                    *in_ as i64,
                    *out_ as i64,
                    config.linear_config,
                )
            })
            .collect();

        Self {
            layers,
            activation: config.activation.maybe_function(),
            output_activation: config.output_activation.maybe_function(),
        }
    }
}

impl Module for Mlp {
    fn has_cudnn_second_derivatives(&self) -> bool {
        self.layers.iter().all(Linear::has_cudnn_second_derivatives)
    }
}

impl FeedForwardModule for Mlp {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut iter_layers = self.layers.iter();
        let mut hidden = iter_layers
            .next()
            .expect("must have >= 1 layers by construction")
            .forward(input);
        for layer in iter_layers {
            if let Some(activation) = self.activation {
                hidden = activation(&hidden);
            }
            hidden = layer.forward(&hidden);
        }
        if let Some(output_activation) = self.output_activation {
            hidden = output_activation(&hidden);
        }
        hidden
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{testing, AsSeq};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{kind::Kind, Device};

    #[fixture]
    fn default_module() -> (Mlp, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = MlpConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let module = config.build_module(&vs.root(), in_dim, out_dim);
        (module, in_dim, out_dim)
    }

    #[rstest]
    fn default_module_forward_batch(default_module: (Mlp, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_forward(&default_mlp, in_dim, out_dim, &[4], Kind::Float);
    }

    #[rstest]
    fn default_module_seq_serial(default_module: (Mlp, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_seq_serial(&AsSeq::new(default_mlp), in_dim, out_dim);
    }

    #[rstest]
    fn default_module_step(default_module: (Mlp, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_step(&AsSeq::new(default_mlp), in_dim, out_dim);
    }

    #[test]
    fn default_module_forward_gradient_descent() {
        testing::check_config_forward_gradient_descent(&MlpConfig::default());
    }
}
