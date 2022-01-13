//! Sequence modules. Like [`tch::nn::Module`] but operate on sequences of data.
pub mod module;
mod rnn;
mod stacked;
#[cfg(test)]
pub mod testing;

pub use rnn::{Gru, GruConfig, Lstm, LstmConfig};
pub use stacked::{Stacked, StackedConfig};

use super::modules::{BuildModule, MlpConfig};
use tch::{IndexOp, Tensor};

pub type GruMlpConfig = StackedConfig<GruConfig, MlpConfig>;
pub type GruMlp = <GruMlpConfig as BuildModule>::Module;
pub type LstmMlpConfig = StackedConfig<LstmConfig, MlpConfig>;
pub type LstmMlp = <LstmMlpConfig as BuildModule>::Module;

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
    ///     A i64 tensor of shape `[MAX_SEQ_LENGTH]`. **Must be on the CPU.**
    ///     Must be monotonically decreasing and positive.
    ///
    /// # Returns
    /// Packed output sequences in the same order as `inputs`.
    /// A tensor of shape `[TOTAL_STEPS, NUM_OUTPUT_FEATURES]`.
    ///
    /// # Panics
    /// Panics if:
    /// * `inputs` device does not match the model device
    /// * `inputs` `NUM_INPUT_FEATURES` dimension does not match the model input features
    /// * `inputs` `TOTAL_STEPS` dimension does not match the sum of `batch_size`
    /// * `batch_sizes` device is not CPU
    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor;
}

box_impl_sequence_module!(dyn SequenceModule + Send);

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
