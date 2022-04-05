mod rnn;

pub use rnn::{Gru, GruConfig, Lstm, LstmConfig};

use tch::{IndexOp, Tensor};

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
