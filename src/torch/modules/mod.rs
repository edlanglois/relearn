//! Torch modules
mod activations;
mod builder;
mod mlp;

pub use activations::Activation;
pub use builder::ModuleBuilder;
pub use mlp::MlpConfig;

use crate::torch::backends::CudnnSupport;
use tch::nn::Module;

impl<M: Module> CudnnSupport for M {
    fn has_cudnn_second_derivatives(&self) -> bool {
        true
    }
}
