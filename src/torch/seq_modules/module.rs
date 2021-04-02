//! Implement sequence module traits for feed-forward modules.
use super::{IterativeModule, SequenceModule};
use tch::{nn::Module, Tensor};

impl<M: Module> SequenceModule for M {
    fn seq_serial(&self, inputs: &Tensor, _seq_lengths: &[usize]) -> Tensor {
        inputs.apply(self)
    }
}

impl<M: Module> IterativeModule for M {
    type State = ();

    fn initial_state(&self, _batch_size: usize) -> Self::State {
        ()
    }

    fn step(&self, input: &Tensor, _state: &Self::State) -> (Tensor, Self::State) {
        (input.apply(self), ())
    }
}
