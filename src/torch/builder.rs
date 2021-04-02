use std::borrow::Borrow;
use tch::nn::Path;

/// Construct an instance of a torch module (or module-like).
pub trait ModuleBuilder {
    type Module;

    /// Build a new module instance.
    ///
    /// # Args
    /// * `vs` - Variable store and namespace.
    /// * `input_dim` - Number of input feature dimensions.
    /// * `output_dim` - Nuber of output feature dimensions.
    fn build<'a, T: Borrow<Path<'a>>>(
        &self,
        vs: T,
        input_dim: usize,
        output_dim: usize,
    ) -> Self::Module;
}
