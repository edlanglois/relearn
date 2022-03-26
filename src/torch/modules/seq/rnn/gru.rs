//! Gated Recurrent Unit
use super::super::super::SequenceModule;
use super::super::seq_serial_map;
use super::{RnnBase, RnnBaseConfig, RnnImpl, RnnLayerWeights};
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, Tensor};

/// Configuration for [`Gru`]
pub type GruConfig = RnnBaseConfig<GruImpl>;

/// Gated recurrent unit module
pub type Gru = RnnBase<GruImpl>;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GruImpl;
impl RnnImpl for GruImpl {
    type CellState = Tensor;

    const CUDNN_MODE: u32 = 3;
    const GATES_MULTIPLE: usize = 3;

    fn initial_cell_state(rnn: &RnnBase<Self>, batch_size: usize) -> Self::CellState {
        Tensor::zeros(
            &[batch_size as i64, rnn.hidden_size as i64],
            (Kind::Float, rnn.device),
        )
    }

    fn cell_batch_step(
        _: &RnnBase<Self>,
        state: &mut Self::CellState,
        w: &RnnLayerWeights,
        batch_input: &Tensor,
    ) -> Tensor {
        *state = batch_input.gru_cell(state, w.w_ih(), w.w_hh(), w.b_ih(), w.b_hh());
        state.shallow_clone()
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
        let batch_size: i64 = shape[0] as i64;
        let num_layers: i64 = self.weights.num_layers() as i64;
        let initial_state = Tensor::zeros(
            &[num_layers, batch_size, self.hidden_size as i64],
            (inputs.kind(), inputs.device()),
        );
        seq_serial_map(inputs, seq_lengths, |seq_input| {
            let (seq_output, _) = seq_input.gru(
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
        // Otherwise torch will segfault
        // See https://github.com/pytorch/pytorch/issues/59418
        assert_eq!(
            batch_sizes.device(),
            Device::Cpu,
            "batch_sizes must be on the CPU"
        );
        let initial_batch_size: i64 = batch_sizes.i(0).into();
        let num_layers: i64 = self.weights.num_layers() as i64;
        let initial_state = Tensor::zeros(
            &[num_layers, initial_batch_size, self.hidden_size as i64],
            (inputs.kind(), inputs.device()),
        );
        let (outputs, _) = Tensor::gru_data(
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
// Confusion with rstest hack when passing the _runner arg
#[allow(
    clippy::needless_pass_by_value,
    clippy::used_underscore_binding,
    clippy::no_effect_underscore_binding
)]
mod tests {
    use super::super::super::super::testing::{
        self, RunIterStep, RunModule, RunSeqPacked, RunSeqSerial,
    };
    use super::super::super::super::Module;
    use super::*;
    use rstest::{fixture, rstest};
    use tch::Device;

    #[fixture]
    fn gru() -> (Gru, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let gru = Gru::new(in_dim, out_dim, Device::Cpu, &GruConfig::default());
        (gru, in_dim, out_dim)
    }

    #[rstest]
    fn seq_serial(gru: (Gru, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_serial(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn seq_packed(gru: (Gru, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_packed(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn seq_step(gru: (Gru, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_step(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn seq_packed_matches_iter_steps(gru: (Gru, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_packed_matches_iter_steps(&gru, in_dim, out_dim);
    }

    #[test]
    fn seq_packed_matches_iter_steps_2layers() {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let config = GruConfig {
            num_layers: 2,
            ..GruConfig::default()
        };
        let gru = Gru::new(in_dim, out_dim, Device::Cpu, &config);
        testing::check_seq_packed_matches_iter_steps(&gru, in_dim, out_dim);
    }

    #[test]
    fn seq_packed_matches_iter_steps_nobias() {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let config = GruConfig {
            bias_init: None,
            ..GruConfig::default()
        };
        let gru = Gru::new(in_dim, out_dim, Device::Cpu, &config);
        testing::check_seq_packed_matches_iter_steps(&gru, in_dim, out_dim);
    }

    #[rstest]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn gradient_descent<R: RunModule<Gru>>(#[case] _runner: R) {
        testing::check_config_gradient_descent::<R, _>(&GruConfig::default());
    }

    #[rstest]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn clone_to_new_device<R: RunModule<Gru>>(#[case] _runner: R) {
        testing::check_config_clone_to_new_device::<R, _>(&GruConfig::default());
    }

    #[test]
    fn clone_to_same_device() {
        testing::check_config_clone_to_same_device::<RunSeqPacked, _>(&GruConfig::default());
    }

    #[rstest]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn ser_de_matches<R: RunModule<Gru>>(#[case] _runner: R, gru: (Gru, usize, usize)) {
        let (module, in_dim, _) = gru;
        testing::check_ser_de_matches::<R, _>(&module, in_dim);
    }

    #[rstest]
    fn variables_count(gru: (Gru, usize, usize)) {
        let (gru, _, _) = gru;
        assert_eq!(gru.variables().count(), 4);
    }

    #[rstest]
    fn trainable_variables_count(gru: (Gru, usize, usize)) {
        let (gru, _, _) = gru;
        assert_eq!(gru.trainable_variables().count(), 4);
    }
}
