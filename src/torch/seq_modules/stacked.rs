use super::{IterativeModule, SequenceModule};
use tch::{nn::Module, Tensor};

/// Append a transformation step to a model.
pub struct PostTransform<T, M: Module> {
    main: T,
    post: M,
}

impl<T: SequenceModule, M: Module> SequenceModule for PostTransform<T, M> {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        self.main.seq_serial(inputs, seq_lengths).apply(&self.post)
    }
}

impl<T: IterativeModule, M: Module> IterativeModule for PostTransform<T, M> {
    type State = T::State;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        self.main.initial_state(batch_size)
    }

    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        let (main_output, state) = self.main.step(input, state);
        let output = main_output.apply(&self.post);
        (output, state)
    }
}
