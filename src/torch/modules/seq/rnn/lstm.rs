//! Long Short-Term Memory
use super::super::super::SequenceModule;
use super::super::seq_serial_map;
use super::{RnnBase, RnnBaseConfig, RnnImpl, RnnLayerWeights};
use tch::{Device, IndexOp, Kind, Tensor};

/// Configuration for [`Lstm`]
pub type LstmConfig = RnnBaseConfig<LstmImpl>;

/// Long Short-Term Memory Module
pub type Lstm = RnnBase<LstmImpl>;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LstmImpl;
impl RnnImpl for LstmImpl {
    type CellState = (Tensor, Tensor);

    const CUDNN_MODE: u32 = 2;
    const GATES_MULTIPLE: u32 = 4;

    fn initial_cell_state(rnn: &RnnBase<Self>, batch_size: i64) -> Self::CellState {
        let hidden_state = Tensor::zeros(&[batch_size, rnn.hidden_size], (Kind::Float, rnn.device));

        let cell_state = hidden_state.shallow_clone();
        (hidden_state, cell_state)
    }

    fn cell_batch_step(
        _: &RnnBase<Self>,
        state: &mut Self::CellState,
        w: &RnnLayerWeights,
        batch_input: &Tensor,
    ) -> Tensor {
        let (ref hidden_state, ref cell_state) = state;
        let (new_hidden_state, new_cell_state) = batch_input.lstm_cell(
            &[hidden_state, cell_state],
            w.w_ih(),
            w.w_hh(),
            w.b_ih(),
            w.b_hh(),
        );
        *state = (new_hidden_state, new_cell_state);
        state.0.shallow_clone()
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
        let num_layers = self.weights.num_layers() as i64;
        let batch_size = shape[0] as i64;
        let zeros = Tensor::zeros(
            &[num_layers, batch_size, self.hidden_size],
            (inputs.kind(), inputs.device()),
        );
        let initial_state = [zeros.shallow_clone(), zeros];

        seq_serial_map(inputs, seq_lengths, |seq_input| {
            let (seq_output, _, _) = seq_input.lstm(
                &initial_state,
                self.weights.flat_weights(),
                self.weights.has_biases,
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
        if batch_sizes.device() != Device::Cpu {
            // Panic here to prevent torch from segfaulting.
            // See https://github.com/pytorch/pytorch/issues/59418
            panic!("Batch sizes must be on the CPU");
        }
        let initial_batch_size: i64 = batch_sizes.i(0).into();
        let num_layers: i64 = self.weights.num_layers() as i64;
        let zeros = Tensor::zeros(
            &[num_layers, initial_batch_size, self.hidden_size],
            (inputs.kind(), inputs.device()),
        );
        let initial_state = [zeros.shallow_clone(), zeros];

        let (outputs, _, _) = Tensor::lstm_data(
            inputs,
            batch_sizes,
            &initial_state,
            self.weights.flat_weights(),
            self.weights.has_biases,
            num_layers,
            self.dropout,
            true,  // train
            false, // bidirectional
        );
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::super::{testing, Module};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    #[fixture]
    fn lstm() -> (Lstm, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let lstm = Lstm::new(&vs.root(), in_dim, out_dim, &LstmConfig::default());
        (lstm, in_dim, out_dim)
    }

    #[rstest]
    fn seq_serial(lstm: (Lstm, usize, usize)) {
        let (lstm, in_dim, out_dim) = lstm;
        testing::check_seq_serial(&lstm, in_dim, out_dim);
    }

    #[rstest]
    fn seq_packed(lstm: (Lstm, usize, usize)) {
        let (lstm, in_dim, out_dim) = lstm;
        testing::check_seq_packed(&lstm, in_dim, out_dim);
    }

    #[rstest]
    fn seq_step(lstm: (Lstm, usize, usize)) {
        let (lstm, in_dim, out_dim) = lstm;
        testing::check_step(&lstm, in_dim, out_dim);
    }

    #[rstest]
    fn seq_packed_matches_iter_steps(lstm: (Lstm, usize, usize)) {
        let (lstm, in_dim, out_dim) = lstm;
        testing::check_seq_packed_matches_iter_steps(&lstm, in_dim, out_dim);
    }

    #[test]
    fn seq_packed_matches_iter_steps_2layers() {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let config = LstmConfig {
            num_layers: 2,
            ..LstmConfig::default()
        };
        let lstm = Lstm::new(&vs.root(), in_dim, out_dim, &config);
        testing::check_seq_packed_matches_iter_steps(&lstm, in_dim, out_dim);
    }

    #[test]
    fn seq_packed_matches_iter_steps_nobias() {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let config = LstmConfig {
            bias_init: None,
            ..LstmConfig::default()
        };
        let lstm = Lstm::new(&vs.root(), in_dim, out_dim, &config);
        testing::check_seq_packed_matches_iter_steps(&lstm, in_dim, out_dim);
    }

    #[test]
    fn seq_packed_gradient_descent() {
        let config = LstmConfig::default();
        testing::check_config_seq_packed_gradient_descent(&config);
    }

    #[test]
    fn clone_to_new_device() {
        testing::check_config_seq_packed_clone_to_new_device(&LstmConfig::default());
    }

    #[test]
    fn clone_to_same_device() {
        testing::check_config_seq_packed_clone_to_same_device(&LstmConfig::default());
    }

    #[rstest]
    fn variables_count(lstm: (Lstm, usize, usize)) {
        let (lstm, _, _) = lstm;
        assert_eq!(lstm.variables().count(), 4);
    }

    #[rstest]
    fn trainable_variables_count(lstm: (Lstm, usize, usize)) {
        let (lstm, _, _) = lstm;
        assert_eq!(lstm.trainable_variables().count(), 4);
    }
}
