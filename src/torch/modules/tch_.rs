//! Module implementations for [`tch`] modules
use super::{BuildModule, FeedForwardModule, Module};
use tch::{nn::Path, Tensor};

impl BuildModule for tch::nn::LinearConfig {
    type Module = tch::nn::Linear;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        tch::nn::linear(
            vs,
            in_dim.try_into().unwrap(),
            out_dim.try_into().unwrap(),
            *self,
        )
    }
}

/// Implement [`Module`] and [`FeedForwardModule`] for a type that implements [`tch::nn::Module`].
macro_rules! impl_ff_module_for_tch {
    ($ty:ty) => {
        impl Module for $ty {}
        impl FeedForwardModule for $ty {
            fn forward(&self, input: &Tensor) -> Tensor {
                tch::nn::Module::forward(self, input)
            }
        }
    };
}
impl_ff_module_for_tch!(tch::nn::Embedding);
impl_ff_module_for_tch!(tch::nn::LayerNorm);
impl_ff_module_for_tch!(tch::nn::Linear);
