//! Iterator modules with self-managed state.
use super::IterativeModule;
use std::borrow::Borrow;
use tch::Tensor;

/// A module that produces iterators over sequences of tensors.
pub trait IterMapModule<'a> {
    type IterMap: TensorIterMap;
    fn iter_map(&'a self, batch_size: usize) -> Self::IterMap;
}

/// An iterative map over a sequence of tensors.
pub trait TensorIterMap {
    /// Apply one step of the iterative map.
    ///
    /// # Args
    /// * `input` - The input for one (batched) step.
    ///     A tensor with shape [BATCH_SIZE, NUM_INPUT_FEATURES]
    ///
    /// # Returns
    /// The output tensor. Has shape [BATCH_SIZE, NUM_OUT_FEATURES]
    fn step(&mut self, input: &Tensor) -> Tensor;
}

/// An encapsulated iterative module with state.
pub struct IterModMap<T: Borrow<M>, M: IterativeModule> {
    iter_mod: T,
    state: M::State,
}
impl<T: Borrow<M>, M: IterativeModule> IterModMap<T, M> {
    fn new(iter_mod: T, batch_size: usize) -> Self {
        let state = iter_mod.borrow().initial_state(batch_size);
        Self { iter_mod, state }
    }
}
impl<T: Borrow<M>, M: IterativeModule> TensorIterMap for IterModMap<T, M> {
    fn step(&mut self, input: &Tensor) -> Tensor {
        let (output, next_state) = self.iter_mod.borrow().step(input, &self.state);
        self.state = next_state;
        output
    }
}
impl<'a, M: IterativeModule + 'a> IterMapModule<'a> for M {
    type IterMap = IterModMap<&'a M, M>;
    fn iter_map(&'a self, batch_size: usize) -> Self::IterMap {
        Self::IterMap::new(self, batch_size)
    }
}
