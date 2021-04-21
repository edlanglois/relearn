//! Long Short-Term Memory
use super::super::{seq_serial_map, IterativeModule, SequenceModule};
use super::{initialize_rnn_params, CudnnRnnMode};
use tch::{nn::Path, Device, Kind, Tensor};

/// A Single-layer Long Short-Term Memory Network.
#[derive(Debug)]
pub struct Lstm {
    // [w_ih, w_hh] or
    // [w_ih, w_hh, b_ih, b_hh]
    params: Vec<Tensor>,
    hidden_size: i64,
    pub dropout: f64,
    device: Device,
}

impl Lstm {
    pub fn new(vs: &Path, in_dim: usize, out_dim: usize, bias: bool, dropout: f64) -> Self {
        let (params, hidden_size, device) =
            initialize_rnn_params(vs, CudnnRnnMode::LSTM, in_dim, out_dim, bias);
        Self {
            params,
            hidden_size,
            dropout,
            device,
        }
    }

    pub fn w_ih(&self) -> &Tensor {
        &self.params[0]
    }
    pub fn w_hh(&self) -> &Tensor {
        &self.params[1]
    }
    pub fn b_ih(&self) -> Option<&Tensor> {
        self.params.get(2)
    }
    pub fn b_hh(&self) -> Option<&Tensor> {
        self.params.get(3)
    }
}

impl IterativeModule for Lstm {
    type State = (Tensor, Tensor);

    fn initial_state(&self, batch_size: usize) -> Self::State {
        let hidden_state = Tensor::zeros(
            &[batch_size as i64, self.hidden_size],
            (Kind::Float, self.device),
        );
        let cell_state = hidden_state.shallow_clone();
        (hidden_state, cell_state)
    }

    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        let (ref hidden_state, ref cell_state) = state;
        let (new_hidden_state, new_cell_state) = input.lstm_cell(
            &[hidden_state, cell_state],
            self.w_ih(),
            self.w_hh(),
            self.b_ih(),
            self.b_hh(),
        );
        let output = new_hidden_state.shallow_clone();
        (output, (new_hidden_state, new_cell_state))
    }
}

impl SequenceModule for Lstm {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        let shape = inputs.size();
        assert_eq!(
            shape.len(),
            3,
            "Input must have 3 dimensions: [BATCH_SIZE, SEQ_LEN, NUM_FEATURES]"
        );
        let batch_size = shape[0];
        let zeros = Tensor::zeros(
            &[1, batch_size as i64, self.hidden_size],
            (inputs.kind(), inputs.device()),
        );
        let initial_state = [zeros.shallow_clone(), zeros];

        let has_biases = self.params.len() > 2;
        seq_serial_map(inputs, seq_lengths, |seq_input| {
            let (seq_output, _, _) = seq_input.lstm(
                &initial_state,
                &self.params,
                has_biases,
                1, // num_layers
                self.dropout,
                true,  // train
                false, // bidirectional
                true,  // batch_first
            );
            seq_output
        })
    }
}

#[cfg(test)]
mod lstm {
    use super::super::super::testing;
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    #[fixture]
    fn lstm() -> (Lstm, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let lstm = Lstm::new(&vs.root(), in_dim, out_dim, true, 0.0);
        (lstm, in_dim, out_dim)
    }

    #[rstest]
    fn lstm_seq_serial(lstm: (Lstm, usize, usize)) {
        let (lstm, in_dim, out_dim) = lstm;
        testing::check_seq_serial(&lstm, in_dim, out_dim);
    }

    #[rstest]
    fn lstm_step(lstm: (Lstm, usize, usize)) {
        let (lstm, in_dim, out_dim) = lstm;
        testing::check_step(&lstm, in_dim, out_dim);
    }
}
