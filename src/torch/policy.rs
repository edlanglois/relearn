//! Torch policy module

use super::{
    backends::CudnnSupport,
    seq_modules::{SequenceModule, StatefulIterativeModule},
};
use tch::Tensor;

/// Policy module
///
/// A policy is a neural network on sequences that supports both batch and iterative processing.
pub trait Policy: SequenceModule + StatefulIterativeModule + CudnnSupport {}
impl<T: SequenceModule + StatefulIterativeModule + CudnnSupport + ?Sized> Policy for T {}

box_impl_sequence_module!(dyn Policy);
box_impl_stateful_iterative_module!(dyn Policy);
box_impl_cudnn_support!(dyn Policy);
