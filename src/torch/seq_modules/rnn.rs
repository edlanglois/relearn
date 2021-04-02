//! Implement sequence module traits for Torch RNNs
use super::{IterativeModule, SequenceModule};
use tch::{nn::RNN, IndexOp, Tensor};

/// Wrapper that implements the Sequence/Iterative Module tratis for [tch::nn::RNN].
struct SeqModRNN<R: RNN>(R);

impl<R: RNN> SequenceModule for SeqModRNN<R> {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        Tensor::cat(
            &seq_lengths
                .into_iter()
                .scan(0, |offset, &length| {
                    let ilength = length as i64;
                    let (output, _) = self
                        .0
                        .seq(&inputs.i((.., *offset..(*offset + ilength), ..)));
                    *offset += ilength;
                    Some(output)
                })
                .collect::<Vec<_>>(),
            -2,
        )
    }
}

impl<R: RNN> IterativeModule for SeqModRNN<R> {
    type State = R::State;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        self.0.zero_state(batch_size as i64)
    }

    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        // (Un)squeeze is to add/remove a seq_len dimension.
        let (output, state) = self.0.seq_init(&input.unsqueeze(-2), state);
        (output.squeeze1(-2), state)
    }
}

impl<R: RNN> From<R> for SeqModRNN<R> {
    fn from(rnn: R) -> Self {
        SeqModRNN(rnn)
    }
}
