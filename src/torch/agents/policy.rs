//! Policy modules
use crate::torch::modules::{BuildModule, IterativeModule, SequenceModule};
use tch::Device;

pub trait Policy: SequenceModule + IterativeModule {}

impl<T: SequenceModule + IterativeModule + ?Sized> Policy for T {}

pub trait BuildPolicy {
    type Policy: Policy;

    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy;
}

impl<T> BuildPolicy for T
where
    T: BuildModule,
    T::Module: SequenceModule + IterativeModule,
{
    type Policy = T::Module;

    #[inline]
    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy {
        self.build_module(in_dim, out_dim, device)
    }
}
