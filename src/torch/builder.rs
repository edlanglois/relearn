use tch::nn::Path;

/// Build an instance of a torch module (or module-like).
pub trait ModuleBuilder<T> {
    /// Build a new module instance.
    ///
    /// # Args
    /// * `vs` - Variable store and namespace.
    /// * `in_dim` - Number of input feature dimensions.
    /// * `out_dim` - Nuber of output feature dimensions.
    fn build(&self, vs: &Path, in_dim: usize, out_dim: usize) -> T;
}
