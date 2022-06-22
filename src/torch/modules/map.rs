use super::{AsModule, Forward, PackedTensor, SeqIterative, SeqPacked, SeqSerial};
use serde::{Deserialize, Serialize};
use tch::Tensor;

/// A module that applies a batch `Tensor -> Tensor` function to the output of another module.
///
/// The function should preserve the batch shape of the input tensor.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchMap<T, F> {
    inner: T,
    f: F,
}

impl<T, F> BatchMap<T, F> {
    pub const fn new(inner: T, f: F) -> Self {
        Self { inner, f }
    }
}

impl<T: AsModule, F> AsModule for BatchMap<T, F> {
    type Module = T::Module;

    fn as_module(&self) -> &Self::Module {
        self.inner.as_module()
    }
    fn as_module_mut(&mut self) -> &mut Self::Module {
        self.inner.as_module_mut()
    }
}

impl<T, F> Forward for BatchMap<T, F>
where
    T: Forward,
    F: Fn(Tensor) -> Tensor,
{
    fn forward(&self, input: &Tensor) -> Tensor {
        (self.f)(self.inner.forward(input))
    }
}

impl<T, F> SeqSerial for BatchMap<T, F>
where
    T: SeqSerial,
    F: Fn(Tensor) -> Tensor,
{
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        (self.f)(self.inner.seq_serial(inputs, seq_lengths))
    }
}

impl<T, F> SeqPacked for BatchMap<T, F>
where
    T: SeqPacked,
    F: Fn(Tensor) -> Tensor,
{
    fn seq_packed(&self, inputs: &PackedTensor) -> PackedTensor {
        self.inner.seq_packed(inputs).batch_map(&self.f)
    }
}

impl<T, F> SeqIterative for BatchMap<T, F>
where
    T: SeqIterative,
    F: Fn(Tensor) -> Tensor,
{
    type State = T::State;
    fn initial_state(&self) -> Self::State {
        self.inner.initial_state()
    }
    fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor {
        (self.f)(self.inner.step(state, input))
    }
}
