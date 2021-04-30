//! Gated Recurrent Unit
use super::super::{seq_serial_map, IterativeModule, SequenceModule};
use super::{initialize_rnn_params, CudnnRnnMode};
use tch::{nn::Path, Device, IndexOp, Kind, Tensor};

/// A single-layer Gated Recurrent Unit Network
#[derive(Debug)]
pub struct Gru {
    // [w_ih, w_hh] or
    // [w_ih, w_hh, b_ih, b_hh]
    params: Vec<Tensor>,
    hidden_size: i64,
    pub dropout: f64,
    device: Device,
}

impl Gru {
    pub fn new(vs: &Path, in_dim: usize, out_dim: usize, bias: bool, dropout: f64) -> Self {
        let (params, hidden_size, device) =
            initialize_rnn_params(vs, CudnnRnnMode::GRU, in_dim, out_dim, bias);
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

impl IterativeModule for Gru {
    type State = Tensor;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        Tensor::zeros(
            &[batch_size as i64, self.hidden_size],
            (Kind::Float, self.device),
        )
    }

    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        let new_state = input.gru_cell(state, self.w_ih(), self.w_hh(), self.b_ih(), self.b_hh());
        (new_state.shallow_clone(), new_state)
    }
}

impl SequenceModule for Gru {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        let shape = inputs.size();
        assert_eq!(
            shape.len(),
            3,
            "Input must have 3 dimensions: [BATCH_SIZE, SEQ_LEN, NUM_FEATURES]"
        );
        let batch_size = shape[0];
        let num_layers = 1;
        let initial_state = Tensor::zeros(
            &[num_layers, batch_size as i64, self.hidden_size],
            (inputs.kind(), inputs.device()),
        );
        let has_biases = self.params.len() > 2;
        seq_serial_map(inputs, seq_lengths, |seq_input| {
            let (seq_output, _) = seq_input.gru(
                &initial_state,
                &self.params,
                has_biases,
                num_layers,
                self.dropout,
                true,  // train
                false, // bidirectional
                true,  // batch_first
            );
            seq_output
        })
    }

    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor {
        let initial_batch_size: i64 = batch_sizes.i(0).into();
        let num_layers = 1;
        let initial_state = Tensor::zeros(
            &[num_layers, initial_batch_size, self.hidden_size],
            (inputs.kind(), inputs.device()),
        );
        let has_biases = self.params.len() > 2;
        let (outputs, _) = Tensor::gru1(
            inputs,
            batch_sizes,
            &initial_state,
            &self.params,
            has_biases,
            num_layers,
            self.dropout,
            true,  // train
            false, // bidirectional
        );
        outputs
    }
}

#[cfg(test)]
mod gru {
    use super::super::super::testing;
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    #[fixture]
    fn gru() -> (Gru, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let gru = Gru::new(&vs.root(), in_dim, out_dim, true, 0.0);
        (gru, in_dim, out_dim)
    }

    #[rstest]
    fn gru_seq_serial(gru: (Gru, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_serial(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_seq_packed(gru: (Gru, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_packed(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_step(gru: (Gru, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_step(&gru, in_dim, out_dim);
    }
}