use super::{IterativeModule, SequenceModule};
use tch::{nn::Func, nn::Module, Tensor};

/// A module stacked on top of a sequence.
pub struct Stacked<'a, T, U> {
    /// The sequence module.
    pub seq: T,
    /// An optional activation function in between
    pub activation: Option<Func<'a>>,
    /// A module applied to the sequence module outputs.
    pub top: U,
}

impl<'a, T, U> Stacked<'a, T, U> {
    pub fn new(seq: T, activation: Option<Func<'a>>, top: U) -> Self {
        Self {
            seq,
            activation,
            top,
        }
    }
}

impl<'a, T, U> SequenceModule for Stacked<'a, T, U>
where
    T: SequenceModule,
    U: Module,
{
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        let mut data = self.seq.seq_serial(inputs, seq_lengths);
        if let Some(ref m) = self.activation {
            data = data.apply(m);
        }
        data.apply(&self.top)
    }
}

impl<'a, T, U> IterativeModule for Stacked<'a, T, U>
where
    T: IterativeModule,
    U: Module,
{
    type State = T::State;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        self.seq.initial_state(batch_size)
    }

    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        let (mut data, state) = self.seq.step(input, state);
        if let Some(ref m) = self.activation {
            data = data.apply(m)
        }
        data = data.apply(&self.top);
        (data, state)
    }
}

#[cfg(test)]
mod sequence_regressor {
    use super::super::{testing, SeqModRnn};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    type GruMlp = Stacked<'static, SeqModRnn<nn::GRU>, nn::Linear>;

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
        let sr = Stacked::new(gru, Some(nn::func(Tensor::relu)), linear);
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
