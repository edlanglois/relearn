//! Torch policy module
use super::{backends::CudnnSupport, modules::BuildModule, seq_modules::SequenceModule};
use tch::nn::Path;
use tch::Tensor;

/// Policy module
pub trait Policy: SequenceModule + CudnnSupport {}
impl<T: SequenceModule + CudnnSupport + ?Sized> Policy for T {}

box_impl_sequence_module!(dyn Policy);
box_impl_cudnn_support!(dyn Policy);
box_impl_sequence_module!(dyn Policy + Send);
box_impl_cudnn_support!(dyn Policy + Send);

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
    B::Module: Policy,
{
    type Policy = B::Module;

    fn build_policy(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Policy {
        self.build_module(vs, in_dim, out_dim)
    }
}
