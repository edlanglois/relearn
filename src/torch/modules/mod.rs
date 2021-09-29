//! Torch modules
mod activations;
mod mlp;

pub use activations::Activation;
pub use mlp::MlpConfig;

use crate::torch::backends::CudnnSupport;
use tch::nn::{Module, Path};

impl<M: Module> CudnnSupport for M {
    fn has_cudnn_second_derivatives(&self) -> bool {
        true
    }
}

/// Build an instance of a torch module (or module-like).
pub trait BuildModule {
    type Module;

    /// Build a new module instance.
    ///
    /// # Args
    /// * `vs` - Variable store and namespace.
    /// * `in_dim` - Number of input feature dimensions.
    /// * `out_dim` - Nuber of output feature dimensions.
    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module;
}
