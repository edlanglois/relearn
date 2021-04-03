//! Implement sequence module traits for Torch RNNs
use super::{IterativeModule, SequenceModule};
use tch::{nn::RNN, IndexOp, Tensor};

/// Wrapper that implements the Sequence/Iterative Module tratis for [tch::nn::RNN].
pub struct SeqModRNN<R: RNN>(R);

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

#[cfg(test)]
mod sequence_module {
    use super::*;
    use tch::{kind::Kind, nn, Device};

    #[test]
    fn gru_seq_serial() {
        let in_features = 3;
        let out_features = 2;
        let batch_size = 4;
        let vs = nn::VarStore::new(Device::Cpu);
        let gru = SeqModRNN::from(nn::gru(
            &vs.root(),
            in_features,
            out_features,
            Default::default(),
        ));

        let seq_lengths = [1usize, 3, 2];
        let total_seq_length: usize = seq_lengths.iter().sum();
        let inputs = Tensor::ones(
            &[batch_size, total_seq_length as i64, in_features],
            (Kind::Float, Device::Cpu),
        );
        let output = gru.seq_serial(&inputs, &seq_lengths);
        // Check shape
        assert_eq!(
            output.size(),
            vec![batch_size, total_seq_length as i64, out_features]
        );
        // Sequences: 0 | 1 2 3 | 4 5
        // Compare the inner sequences. The RNN should reset for each.
        assert_eq!(output.i((.., 0, ..)), output.i((.., 1, ..)));
        assert_eq!(output.i((.., 1..3, ..)), output.i((.., 4..6, ..)));
    }
}

#[cfg(test)]
mod iterative_module {
    use super::*;
    use tch::{kind::Kind, nn, Device};

    #[test]
    fn gru() {
        let in_features = 3;
        let out_features = 2;
        let batch_size = 4;
        let vs = nn::VarStore::new(Device::Cpu);
        let gru = SeqModRNN::from(nn::gru(
            &vs.root(),
            in_features,
            out_features,
            Default::default(),
        ));

        let state = gru.initial_state(batch_size);
        let input = Tensor::ones(
            &[batch_size as i64, in_features],
            (Kind::Float, Device::Cpu),
        );
        let (output, _) = gru.step(&input, &state);
        assert_eq!(output.size(), vec![batch_size as i64, out_features]);
    }
}
