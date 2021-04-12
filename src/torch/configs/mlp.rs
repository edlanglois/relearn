use super::super::{Activation, ModuleBuilder};
use std::iter;
use tch::nn;

/// Multi-Layer Perceptron Configuration
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlpConfig {
    /// Sizes of the hidden layers
    pub hidden_sizes: Vec<usize>,
    /// Activation function between hidden layers.
    pub activation: Activation,
    /// Activation function on the output.
    pub output_activation: Activation,
}

impl Default for MlpConfig {
    fn default() -> Self {
        MlpConfig {
            hidden_sizes: vec![128],
            activation: Activation::Relu,
            output_activation: Activation::Identity,
        }
    }
}

impl ModuleBuilder<nn::Sequential> for MlpConfig {
    fn build_module(&self, vs: &nn::Path, input_dim: usize, output_dim: usize) -> nn::Sequential {
        let iter_in_dim = iter::once(&input_dim).chain(self.hidden_sizes.iter());
        let iter_out_dim = self.hidden_sizes.iter().chain(iter::once(&output_dim));

        let mut layers = nn::seq();
        for (i, (&layer_in_dim, &layer_out_dim)) in iter_in_dim.zip(iter_out_dim).enumerate() {
            if i > 0 {
                if let Some(m) = self.activation.maybe_module() {
                    layers = layers.add(m);
                }
            }
            layers = layers.add(nn::linear(
                vs / format!("layer_{}", i),
                layer_in_dim as i64,
                layer_out_dim as i64,
                Default::default(),
            ));
        }

        if let Some(m) = self.output_activation.maybe_module() {
            layers = layers.add(m);
        }
        layers
    }
}

#[cfg(test)]
mod mlp_config {
    use super::super::super::seq_modules::testing;
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{kind::Kind, Device};

    type MLP = nn::Sequential;

    #[fixture]
    fn default_module() -> (MLP, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = MlpConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let module = config.build_module(&vs.root(), in_dim, out_dim);
        (module, in_dim, out_dim)
    }

    #[rstest]
    fn default_module_forward_batch(default_module: (MLP, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_forward(&default_mlp, in_dim, out_dim, &[4], Kind::Float);
    }

    #[rstest]
    fn default_module_seq_serial(default_module: (MLP, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_seq_serial(&default_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn default_module_step(default_module: (MLP, usize, usize)) {
        let (default_mlp, in_dim, out_dim) = default_module;
        testing::check_step(&default_mlp, in_dim, out_dim);
    }
}
