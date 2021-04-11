use std::borrow::Borrow;
use tch::nn::Path;

/// Build an instance of a torch module (or module-like).
pub trait ModuleBuilder<T> {
    /// Build a new module instance.
    ///
    /// # Args
    /// * `vs` - Variable store and namespace.
    /// * `input_dim` - Number of input feature dimensions.
    /// * `output_dim` - Nuber of output feature dimensions.
    fn build<'a, P: Borrow<Path<'a>>>(&self, vs: P, input_dim: usize, output_dim: usize) -> T;
}
