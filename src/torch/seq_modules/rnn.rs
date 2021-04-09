//! Implement sequence module traits for Torch RNNs
use super::{IterativeModule, SequenceModule};
use tch::{nn::RNN, IndexOp, Tensor};

/// Wrapper that implements the Sequence/Iterative Module tratis for [tch::nn::RNN].
pub struct SeqModRnn<R: RNN>(R);

impl<R: RNN> SequenceModule for SeqModRnn<R> {
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

impl<R: RNN> IterativeModule for SeqModRnn<R> {
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

impl<R: RNN> From<R> for SeqModRnn<R> {
    fn from(rnn: R) -> Self {
        SeqModRnn(rnn)
    }
}

#[cfg(test)]
mod seq_mod_rnn {
    use super::super::testing;
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    #[fixture]
    fn gru() -> (SeqModRnn<nn::GRU>, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let gru = SeqModRnn::from(nn::gru(
            &vs.root(),
            in_dim as i64,
            out_dim as i64,
            Default::default(),
        ));
        (gru, in_dim, out_dim)
    }

    #[rstest]
    fn gru_seq_serial(gru: (SeqModRnn<nn::GRU>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_serial(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_step(gru: (SeqModRnn<nn::GRU>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_step(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_iter_map(gru: (SeqModRnn<nn::GRU>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_iter_map(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_iter_map_parallel(gru: (SeqModRnn<nn::GRU>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_iter_map_parallel(&gru, in_dim, out_dim);
    }
}
