//! Neural network modules: variables and implementation for a part of a neural network.
mod chain;
mod ff;
mod seq;
#[cfg(test)]
pub mod testing;

pub use chain::{Chain, ChainConfig};
pub use ff::{Activation, Linear, LinearConfig, Mlp, MlpConfig};
pub use seq::{Gru, GruConfig, Lstm, LstmConfig};

pub type GruMlpConfig = ChainConfig<GruConfig, MlpConfig>;
pub type LstmMlpConfig = ChainConfig<GruConfig, MlpConfig>;

use crate::torch::packed::PackedTensor;
use tch::{Device, Tensor};

/// A self-contained part of a neural network.
pub trait Module {
    /// Create a clone of this module sharing the same variables (tensors).
    #[must_use]
    fn shallow_clone(&self) -> Self
    where
        Self: Sized;

    /// Create a clone of this module on the given device.
    ///
    /// A shallow clone is created if the given device is the same as the current device
    /// otherwise the tensors are copied.
    #[must_use]
    fn clone_to_device(&self, device: Device) -> Self
    where
        Self: Sized;

    /// Iterator over variables (tensors) managed by this module.
    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_>;

    /// Iterator over the trainable variables (tensors) managed by this module.
    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_>;

    /// Whether cuDNN supports second derivatives of this module.
    fn has_cudnn_second_derivatives(&self) -> bool {
        true // usually true
    }

    /// Chain another module after this one with an activation function in between.
    fn chain<M>(self, other: M, activation: Activation) -> Chain<Self, M>
    where
        Self: Sized,
    {
        Chain {
            first: self,
            second: other,
            activation,
        }
    }
}

/// Implement `Module` for a deref-able wrapper type generic over `T: Module + ?Sized`.
macro_rules! impl_wrapped_module {
    ($wrapper:ty) => {
        impl<T: Module + ?Sized> Module for $wrapper {
            fn shallow_clone(&self) -> Self {
                // TODO: Get this implemented.
                // Maybe need a boxed_shallow_clone() -> Box<dyn Module> method and
                // replace macro calls with a dedicated impl for Box<dyn Module>.
                unimplemented!()
            }
            fn clone_to_device(&self, _: Device) -> Self {
                unimplemented!()
            }
            fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
                T::variables(self)
            }
            fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
                T::trainable_variables(self)
            }
            fn has_cudnn_second_derivatives(&self) -> bool {
                T::has_cudnn_second_derivatives(&self)
            }
        }
    };
}
impl_wrapped_module!(&'_ T);
impl_wrapped_module!(Box<T>);

/// Extra module methods with a less convenient interface. Prefer using [`Module`] instead.
pub trait ModuleExtras<'a> {
    type Variables: Iterator<Item = &'a Tensor>;
    type TrainableVariables: Iterator<Item = &'a Tensor>;

    /// Iterator over variables (tensors) managed by this module.
    fn variables(&'a self) -> Self::Variables;

    /// Iterator over the trainable variables (tensors) managed by this module.
    fn trainable_variables(&'a self) -> Self::TrainableVariables;
}

/// Build a [`Module`]
pub trait BuildModule {
    type Module: Module;

    /// Build a new module instance.
    ///
    /// # Args
    /// * `in_dim`  - Number of input feature dimensions.
    /// * `out_dim` - Number of output feature dimensions.
    /// * `device`  - Device on which to create the model tensors.
    fn build_module(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Module;

    /// Chain another module configuration after this one with an activation function in between.
    fn chain<MC>(
        self,
        other: MC,
        hidden_dim: usize,
        activation: Activation,
    ) -> ChainConfig<Self, MC>
    where
        Self: Sized,
    {
        ChainConfig {
            first_config: self,
            second_config: other,
            hidden_dim,
            activation,
        }
    }
}

/// Implement [`BuildModule`] for a deref-able wrapper type generic over `T: BuildModule + ?Sized`.
macro_rules! impl_wrapped_build_module {
    ($wrapper:ty) => {
        impl<T: BuildModule + ?Sized> BuildModule for $wrapper {
            type Module = T::Module;

            fn build_module(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Module {
                T::build_module(self, in_dim, out_dim, device)
            }
        }
    };
}
impl_wrapped_build_module!(&'_ T);
impl_wrapped_build_module!(Box<T>);

/// A network module implementing a feed-forward transformation.
///
/// This is roughtly equivalent to PyTorch's [module][module] class except it is not treated as
/// the base interface of all modules because not all modules implement the batch
/// `Tensor -> Tensor` transformation.
///
/// [module]: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
pub trait FeedForwardModule: Module {
    /// Apply a batch feed-forward transformation to a tensor.
    ///
    /// Applies a vector-to-vector map on the last dimension of the input tensor,
    /// replicated over all other dimensions.
    ///
    /// * Input shape: `[BATCH_SHAPE.., INPUT_DIM]`.
    /// * Output shape: `[BATCH_SHAPE.., OUTPUT_DIM]`.
    fn forward(&self, input: &Tensor) -> Tensor;
}

/// Implement [`FeedForwardModule`] for a deref-able generic wrapper type.
macro_rules! impl_wrapped_feed_forward_module {
    ($wrapper:ty) => {
        impl<T: FeedForwardModule + ?Sized> FeedForwardModule for $wrapper {
            fn forward(&self, input: &Tensor) -> Tensor {
                T::forward(self, input)
            }
        }
    };
}
impl_wrapped_feed_forward_module!(&'_ T);
impl_wrapped_feed_forward_module!(Box<T>);

/// A network module implementing a sequence transformation.
pub trait SequenceModule: Module {
    /// Apply the network over multiple sequences arranged in series one after another.
    ///
    /// # Args
    /// * `inputs` - Batched input sequences arranged in series.
    ///     An f32 tensor of shape `[BATCH_SIZE, TOTAL_SEQ_LENGTH, NUM_INPUT_FEATURES]`
    /// * `seq_lengths` - Length of each sequence.
    ///     The sequence length is the same across the batch dimension.
    ///
    /// If `seq_lengths = [L0, L1, ..., LN]` then
    /// `inputs[.., 0..L0, ..]` is the first batch of sequences,
    /// `inputs[.., L0..L1, ..]` is the second, etc.
    ///
    /// # Returns
    /// Batched output sequences arranged in series.
    /// A tensor of shape `[BATCH_SHAPE, TOTAL_SEQ_LENGTH, NUM_OUTPUT_FEATURES]`.
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor;

    /// Apply the network to a [`PackedTensor`]
    ///
    /// # Args
    /// * `inputs` - Packed input sequences.
    ///     An f32 tensor of shape `[TOTAL_STEPS, NUM_INPUT_FEATURES]`
    ///
    /// # Returns
    /// Packed output sequences with the same structure as `inputs`.
    ///
    /// # Panics
    /// Panics if:
    /// * `inputs` device does not match the model device
    /// * `inputs` `NUM_INPUT_FEATURES` dimension does not match the model input features
    fn seq_packed(&self, inputs: &PackedTensor) -> PackedTensor;
}

/// Implement [`SequenceModule`] for a deref-able generic wrapper type.
macro_rules! impl_wrapped_sequence_module {
    ($wrapper:ty) => {
        impl<T: SequenceModule + ?Sized> SequenceModule for $wrapper {
            fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
                T::seq_serial(self, inputs, seq_lengths)
            }
            fn seq_packed(&self, inputs: &PackedTensor) -> PackedTensor {
                T::seq_packed(self, inputs)
            }
        }
    };
}
impl_wrapped_sequence_module!(&'_ T);
impl_wrapped_sequence_module!(Box<T>);

/// A module that operates iteratively on a sequence of data.
pub trait IterativeModule: Module {
    /// Sequence state managed by the module.
    type State;

    /// Construct an initial state for the start of a new sequence.
    fn initial_state(&self) -> Self::State;

    /// Apply one step of the module.
    ///
    /// # Args
    /// * `input` - The input for one step.
    ///     A tensor with shape `[NUM_INPUT_FEATURES]`
    /// * `state` - The policy hidden state.
    ///
    /// # Returns
    /// * `output` - The output tensor. Has shape `[NUM_OUT_FEATURES]`
    /// * `state` - A new value for the hidden state.
    fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor;

    /// Iterate over input tensors
    fn iter<I>(&self, inputs: I) -> ModuleStepIter<&Self, I::IntoIter>
    where
        I: IntoIterator,
        I::Item: AsRef<Tensor>,
    {
        ModuleStepIter::new(self, inputs.into_iter())
    }

    /// Convert into an iterator over input tensors.
    fn into_iter<I>(self, inputs: I) -> ModuleStepIter<Self, I::IntoIter>
    where
        I: IntoIterator,
        I::Item: AsRef<Tensor>,
        Self: Sized,
    {
        ModuleStepIter::new(self, inputs.into_iter())
    }
}

/// Implement [`IterativeModule`] for a deref-able generic wrapper type.
macro_rules! impl_wrapped_iterative_module {
    ($wrapper:ty) => {
        impl<T: IterativeModule + ?Sized> IterativeModule for $wrapper {
            type State = T::State;
            fn initial_state(&self) -> Self::State {
                T::initial_state(self)
            }
            fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor {
                T::step(self, state, input)
            }
        }
    };
}
impl_wrapped_iterative_module!(&'_ T);
impl_wrapped_iterative_module!(Box<T>);

/// Iterator over [`IterativeModule`] step outputs.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ModuleStepIter<M: IterativeModule, I> {
    module: M,
    state: M::State,
    inputs: I,
}

impl<M: IterativeModule, I> ModuleStepIter<M, I> {
    fn new(module: M, inputs: I) -> Self {
        Self {
            state: module.initial_state(),
            module,
            inputs,
        }
    }
}

impl<M, I> Iterator for ModuleStepIter<M, I>
where
    M: IterativeModule,
    I: Iterator,
    I::Item: AsRef<Tensor>,
{
    type Item = Tensor;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(
            self.module
                .step(&mut self.state, self.inputs.next()?.as_ref()),
        )
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inputs.size_hint()
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let module = self.module;
        self.inputs
            .fold(
                (self.state, init),
                move |(mut module_state, fold_state), input| {
                    let new_fold_state =
                        f(fold_state, module.step(&mut module_state, input.as_ref()));
                    (module_state, new_fold_state)
                },
            )
            .1
    }
}

impl<M, I> ExactSizeIterator for ModuleStepIter<M, I>
where
    M: IterativeModule,
    I: ExactSizeIterator,
    I::Item: AsRef<Tensor>,
{
    fn len(&self) -> usize {
        self.inputs.len()
    }
}
