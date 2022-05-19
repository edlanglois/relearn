//! Policy modules
use crate::torch::modules::{BuildModule, SeqIterative, SeqPacked};
use tch::Device;

pub trait Policy: SeqPacked + SeqIterative {}

impl<T: SeqPacked + SeqIterative + ?Sized> Policy for T {}

pub trait BuildPolicy {
    type Policy: Policy;

    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy;
}

impl<T> BuildPolicy for T
where
    T: BuildModule,
    T::Module: SeqPacked + SeqIterative,
{
    type Policy = T::Module;

    #[inline]
    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy {
        self.build_module(in_dim, out_dim, device)
    }
}
