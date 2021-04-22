use super::{SequenceModule, StatefulIterSeqModule, StatefulIterativeModule};
use tch::Tensor;

impl SequenceModule for Box<dyn StatefulIterSeqModule> {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        self.as_ref().seq_serial(inputs, seq_lengths)
    }

    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor {
        self.as_ref().seq_packed(inputs, batch_sizes)
    }
}

impl StatefulIterativeModule for Box<dyn StatefulIterSeqModule> {
    fn step(&mut self, input: &Tensor) -> Tensor {
        self.as_mut().step(input)
    }
    fn reset(&mut self) {
        self.as_mut().reset()
    }
}
