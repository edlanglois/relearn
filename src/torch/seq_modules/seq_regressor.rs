use super::{IterativeModule, SequenceModule};
use tch::{nn::Func, nn::Module, Tensor};

/// A sequence regressor module.
///
/// A sequence module followed by a nonlinearity then a feed-forward module.
pub struct SequenceRegressor<'a, S, M: Module> {
    seq: S,
    activation: Option<Func<'a>>,
    post_transform: M,
}

impl<'a, S, M: Module> SequenceRegressor<'a, S, M> {
    pub fn new(seq: S, activation: Option<Func<'a>>, post_transform: M) -> Self {
        Self {
            seq,
            activation,
            post_transform,
        }
    }
}

impl<'a, S: SequenceModule, M: Module> SequenceModule for SequenceRegressor<'a, S, M> {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        let mut data = self.seq.seq_serial(inputs, seq_lengths);
        if let Some(ref m) = self.activation {
            data = data.apply(m);
        }
        data.apply(&self.post_transform)
    }
}

impl<'a, S: IterativeModule, M: Module> IterativeModule for SequenceRegressor<'a, S, M> {
    type State = S::State;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        self.seq.initial_state(batch_size)
    }

    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        let (mut data, state) = self.seq.step(input, state);
        if let Some(ref m) = self.activation {
            data = data.apply(m)
        }
        data = data.apply(&self.post_transform);
        (data, state)
    }
}

#[cfg(test)]
mod sequence_regressor {
    use super::super::{testing, SeqModRnn};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    type GruMlp = SequenceRegressor<'static, SeqModRnn<nn::GRU>, nn::Linear>;

    /// GRU followed by a relu then a linear layer.
    #[fixture]
    fn gru_relu_linear() -> (GruMlp, usize, usize) {
        let in_dim: usize = 3;
        let hidden_dim: usize = 5;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let path = &vs.root();
        let gru = SeqModRnn::from(nn::gru(
            &(path / "rnn"),
            in_dim as i64,
            hidden_dim as i64,
            Default::default(),
        ));
        let linear = nn::linear(
            &(path / "linear"),
            hidden_dim as i64,
            out_dim as i64,
            Default::default(),
        );
        let sr = SequenceRegressor::new(gru, Some(nn::func(Tensor::relu)), linear);
        (sr, in_dim, out_dim)
    }

    #[rstest]
    fn gru_relu_linear_seq_serial(gru_relu_linear: (GruMlp, usize, usize)) {
        let (gru_relu_linear, in_dim, out_dim) = gru_relu_linear;
        testing::check_seq_serial(&gru_relu_linear, in_dim, out_dim);
    }

    #[rstest]
    fn gru_relu_linear_step(gru_relu_linear: (GruMlp, usize, usize)) {
        let (gru_relu_linear, in_dim, out_dim) = gru_relu_linear;
        testing::check_step(&gru_relu_linear, in_dim, out_dim);
    }
}
