//! Linear layer
use super::super::{
    BuildModule, FeedForwardModule, IterativeModule, Module, ModuleExtras, SequenceModule,
};
use crate::torch::initializers::Initializer;
use std::iter::{self, Chain, Once};
use std::option;
use tch::{nn::Path, Device, Tensor};

/// Configuration for the [`Linear`] module.
#[derive(Debug, Copy, Clone, PartialEq)]
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
#[derive(Debug, PartialEq)]
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

impl Module for Linear {
    fn shallow_clone(&self) -> Self
    where
        Self: Sized,
    {
        Self {
            kernel: self.kernel.shallow_clone(),
            bias: self.bias.as_ref().map(Tensor::shallow_clone),
        }
    }

    fn clone_to_device(&self, device: Device) -> Self
    where
        Self: Sized,
    {
        Self {
            kernel: self.kernel.to_device(device),
            bias: self.bias.as_ref().map(|b| b.to_device(device)),
        }
    }

    #[inline]
    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(ModuleExtras::variables(self))
    }

    #[inline]
    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(ModuleExtras::trainable_variables(self))
    }
}

impl<'a> ModuleExtras<'a> for Linear {
    type Variables = Chain<Once<&'a Tensor>, option::Iter<'a, Tensor>>;
    type TrainableVariables = Self::Variables;

    #[inline]
    fn variables(&'a self) -> Self::Variables {
        iter::once(&self.kernel).chain(self.bias.iter())
    }

    #[inline]
    fn trainable_variables(&'a self) -> Self::TrainableVariables {
        ModuleExtras::variables(self)
    }
}

impl FeedForwardModule for Linear {
    #[inline]
    fn forward(&self, input: &Tensor) -> Tensor {
        input.linear(&self.kernel, self.bias.as_ref())
    }
}

/// Sequence processing by batching over the sequence dimension.
impl SequenceModule for Linear {
    #[inline]
    fn seq_serial(&self, inputs: &Tensor, _seq_lengths: &[usize]) -> Tensor {
        self.forward(inputs)
    }

    #[inline]
    fn seq_packed(&self, inputs: &Tensor, _batch_sizes: &Tensor) -> Tensor {
        self.forward(inputs)
    }
}

/// Iterate over a sequence by independently and identically transforming each step.
impl IterativeModule for Linear {
    type State = ();

    #[inline]
    fn initial_state(&self) -> Self::State {}

    #[inline]
    fn step(&self, _: &mut Self::State, input: &Tensor) -> Tensor {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::testing;
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

    #[fixture]
    fn module_no_bias() -> (Linear, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = LinearConfig {
            bias_init: None,
            ..LinearConfig::default()
        };
        let vs = VarStore::new(Device::Cpu);
        let module = config.build_module(&vs.root(), in_dim, out_dim);
        (module, in_dim, out_dim)
    }

    #[rstest]
    fn forward_batch(default_module: (Linear, usize, usize)) {
        let (module, in_dim, out_dim) = default_module;
        testing::check_forward(&module, in_dim, out_dim, &[4], Kind::Float);
    }

    #[rstest]
    fn seq_serial(default_module: (Linear, usize, usize)) {
        let (module, in_dim, out_dim) = default_module;
        testing::check_seq_serial(&module, in_dim, out_dim);
    }

    #[rstest]
    fn seq_packed(default_module: (Linear, usize, usize)) {
        let (module, in_dim, out_dim) = default_module;
        testing::check_seq_packed(&module, in_dim, out_dim);
    }

    #[rstest]
    fn seq_step(default_module: (Linear, usize, usize)) {
        let (module, in_dim, out_dim) = default_module;
        testing::check_step(&module, in_dim, out_dim);
    }

    #[rstest]
    fn seq_consistent(default_module: (Linear, usize, usize)) {
        let (module, in_dim, out_dim) = default_module;
        testing::check_seq_packed_matches_iter_steps(&module, in_dim, out_dim);
    }

    #[test]
    fn forward_gradient_descent() {
        testing::check_config_forward_gradient_descent(&LinearConfig::default());
    }

    #[test]
    fn seq_packed_gradient_descent() {
        testing::check_config_forward_gradient_descent(&LinearConfig::default());
    }

    #[test]
    fn clone_to_new_device() {
        testing::check_config_forward_clone_to_new_device(&LinearConfig::default());
    }

    #[test]
    fn clone_to_same_device() {
        testing::check_config_forward_clone_to_same_device(&LinearConfig::default());
    }

    #[rstest]
    fn variables_count_default(default_module: (Linear, usize, usize)) {
        let (module, _, _) = default_module;
        assert_eq!(Module::variables(&module).count(), 2);
    }

    #[rstest]
    fn variables_count_no_bias(module_no_bias: (Linear, usize, usize)) {
        let (module, _, _) = module_no_bias;
        assert_eq!(Module::variables(&module).count(), 1);
    }

    #[rstest]
    fn trainable_variables_count_default(default_module: (Linear, usize, usize)) {
        let (module, _, _) = default_module;
        assert_eq!(Module::trainable_variables(&module).count(), 2);
    }

    #[rstest]
    fn trainable_variables_count_no_bias(module_no_bias: (Linear, usize, usize)) {
        let (module, _, _) = module_no_bias;
        assert_eq!(Module::trainable_variables(&module).count(), 1);
    }
}
