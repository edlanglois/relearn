//! Torch policy module
use super::{
    backends::CudnnSupport,
    modules::BuildModule,
    seq_modules::{SequenceModule, StatefulIterativeModule},
};
use tch::nn::Path;
use tch::Tensor;

/// Policy module
///
/// A policy is a neural network on sequences that supports both batch and iterative processing.
pub trait Policy: SequenceModule + StatefulIterativeModule + CudnnSupport {}
impl<T: SequenceModule + StatefulIterativeModule + CudnnSupport + ?Sized> Policy for T {}

box_impl_sequence_module!(dyn Policy);
box_impl_stateful_iterative_module!(dyn Policy);
box_impl_cudnn_support!(dyn Policy);

/// Build a [`Policy`] instance.
pub trait BuildPolicy {
    type Policy: Policy;

    /// Build a new policy instance.
    ///
    /// # Args
    /// * `vs` - Variable store and namespace.
    /// * `in_dim` - Number of input feature dimensions.
    /// * `out_dim` - Nuber of output feature dimensions.
    fn build_policy(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Policy;
}

impl<B> BuildPolicy for B
where
    B: BuildModule + ?Sized,
    <B as BuildModule>::Module: Policy,
{
    type Policy = <Self as BuildModule>::Module;

    fn build_policy(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Policy {
        self.build_module(vs, in_dim, out_dim)
    }
}
