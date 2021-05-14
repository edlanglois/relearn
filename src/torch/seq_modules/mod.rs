//! Sequence modules. Like [`tch::nn::Module`] but operate on sequences of data.
mod r#box;
pub mod module;
mod rnn;
mod stacked;
#[cfg(test)]
pub mod testing;
mod with_state;

pub use rnn::{Gru, Lstm, RnnConfig};
pub use stacked::{Stacked, StackedConfig};
pub use with_state::WithState;

use super::modules::MlpConfig;
use tch::{nn, IndexOp, Tensor};

/// An MLP stacked on top of a GRU.
pub type GruMlp = Stacked<'static, Gru, nn::Sequential>;
/// An MLP stacked on top of an LSTM.
pub type LstmMlp = Stacked<'static, Lstm, nn::Sequential>;
/// Configuration for an MLP stacked on top of an RNN.
pub type RnnMlpConfig = StackedConfig<RnnConfig, MlpConfig>;

/// A network module that operates on a sequence of data.
pub trait SequenceModule {
    /// Apply the network over multiple sequences arranged in series one after another.
    ///
    /// `input.i(.., ..seq_lengths[0], ..)` is the first batch of sequences,
    /// `input.i(.., seq_lengths[0]..(seq_lengths[0]+seq_lengths[1]), ..)` is the second, etc.
    ///
    /// # Args
    /// * `inputs` - Batched input sequences arranged in series.
    ///     An f32 tensor of shape `[BATCH_SIZE, TOTAL_SEQ_LENGTH, NUM_INPUT_FEATURES]`
    /// * `seq_lengths` - Length of each sequence.
    ///     The sequence length is the same across the batch dimension.
    ///
    /// # Returns
    /// Batched output sequences arranged in series.
    /// A tensor of shape `[BATCH_SHAPE, TOTAL_SEQ_LENGTH, NUM_OUTPUT_FEATURES]`.
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor;

    /// Apply the network over multiple sequences packed together in heterogeneous batches.
    ///
    /// `input.i(0..batch_sizes[0], ..)` are the batched first steps of all sequences,
    /// `input.i(batch_sizes[0]..(batch_sizes[0]+batch_sizes[1]), ..)` are the second steps, etc.
    ///
    /// # Args
    /// * `inputs` - Packed input sequences.
    ///     An f32 tensor of shape `[TOTAL_STEPS, NUM_INPUT_FEATURES]`
    ///     where the `TOTAL_STEPS` dimension consists of the packed and batched steps ordered first
    ///     by index within a sequence, then by batch index.
    ///     Sequences must be ordered from longest to shortest.
    ///
    ///     If all sequences have the same length then the `TOTAL_STEPS` dimension
    ///     corresponds to a flattend Tensor of shape `[SEQ_LENGTH, BATCH_SIZE]`.
    ///
    /// * `batch_sizes` - The batch size of each in-sequence step index.
    ///     A i64 tensor of shape `[MAX_SEQ_LENGTH]`.
    ///     Must be monotonically decreasing and positive.
    ///
    /// # Returns
    /// Packed output sequences in the same order as `inputs`.
    /// A tensor of shape `[TOTAL_STEPS, NUM_OUTPUT_FEATURES]`.
    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor;
}

/// A network module that operates iteratively on a sequence of data.
pub trait IterativeModule {
    /// Internal state of the module.
    type State;

    /// Construct an initial state for the start of a new sequence.
    fn initial_state(&self, batch_size: usize) -> Self::State;

    /// Apply one step of the module.
    ///
    /// # Args
    /// * `input` - The input for one (batched) step.
    ///     A tensor with shape `[BATCH_SIZE, NUM_INPUT_FEATURES]`
    /// * `state` - The policy hidden state.
    ///
    /// # Returns
    /// * `output` - The output tensor. Has shape `[BATCH_SIZE, NUM_OUT_FEATURES]`
    /// * `state` - A new value for the hidden state.
    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State);
}

/// A network module that operates iterative on a sequence of data and stores its own state.
pub trait StatefulIterativeModule {
    /// Apply one step of the module.
    ///
    /// # Args
    /// * `input` - The input for one step. A tensor with shape `[NUM_INPUT_FEATURES]`
    ///
    /// # Returns
    /// The output tensor. Has shape `[NUM_OUT_FEATURES]`
    fn step(&mut self, input: &Tensor) -> Tensor;

    /// Reset the inner state for the start of a new sequence.
    ///
    /// It is not necessary to call this at the start of the first sequence.
    fn reset(&mut self);
}

/// Sequence module with stateful iteration
pub trait StatefulIterSeqModule: SequenceModule + StatefulIterativeModule {}
impl<T: SequenceModule + StatefulIterativeModule> StatefulIterSeqModule for T {}

/// Helper function to implement [`SequenceModule::seq_serial`] from a single-sequence closure.
///
/// # Args:
/// * `inputs` - Batched input sequences arranged in series.
///     An f32 tensor of shape `[BATCH_SIZE, TOTAL_SEQ_LENGTH, NUM_INPUT_FEATURES]`
/// * `seq_lengths` - Length of each sequence.
///     The sequence length is the same across the batch dimension.
/// * `f_seq` - A closure that applies a module to a single sequence.
///     Takes an input f32 tensor of shape `[BATCH_SIZE, SEQ_LENGTH, NUM_INPUT_FEATURES]`
///     to an output f32 tensor of shape `[BATCH_SIZE, SEQ_LENGTH, NUM_OUTPUT_FEATURES]`.
///
fn seq_serial_map<F>(inputs: &Tensor, seq_lengths: &[usize], f_seq: F) -> Tensor
where
    F: Fn(Tensor) -> Tensor,
{
    Tensor::cat(
        &seq_lengths
            .iter()
            .scan(0, |offset, &length| {
                let length = length as i64;
                let seq_input = inputs.i((.., *offset..(*offset + length), ..));
                *offset += length;
                Some(f_seq(seq_input))
            })
            .collect::<Vec<_>>(),
        -2,
    )
}
