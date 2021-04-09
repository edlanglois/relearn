//! Sequence modules. Like [tch::nn::Module] but operate on a sequence of data.
mod as_stateful;
pub mod module;
mod rnn;
mod seq_regressor;
#[cfg(test)]
pub mod testing;

pub use as_stateful::AsStatefulIterator;
pub use rnn::SeqModRnn;
pub use seq_regressor::SequenceRegressor;

use tch::Tensor;

/// A network module that operates on a sequence of data.
pub trait SequenceModule {
    /// Apply the network over multiple sequences arranged in series one after another.
    ///
    /// `input.i(.., ..seq_lengths[0], ..)` is the first batch of sequences,
    /// `input.i(.., seq_lengths[0]..seq_lengths[1], ..)` is the second, etc.
    ///
    /// # Args:
    /// * `inputs` - Batched input sequences arranged in series.
    ///     A tensor of shape [BATCH_SIZE, TOTAL_SEQ_LENGTH, NUM_INPUT_FEATURES]
    /// * `seq_lengths` - Length of each sequence.
    ///     The sequence length is the same across the batch dimension.
    ///
    /// # Returns
    /// * `outputs`: Batched output sequences arranged in series.
    ///     A tensor of shape [BATCH_SHAPE, TOTAL_SEQ_LENGTH, NUM_OUTPUT_FEATURES]
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor;

    // TODO: seq_packed
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
    ///     A tensor with shape [BATCH_SIZE, NUM_INPUT_FEATURES]
    /// * `state` - The policy hidden state.
    ///
    /// # Returns
    /// * `output` - The output tensor. Has shape [BATCH_SIZE, NUM_OUT_FEATURES]
    /// * `state` - A new value for the hidden state.
    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State);
}

/// A network module that operates iterative on a sequence of data and stores its own state.
pub trait StatefulIterativeModule {
    /// Apply one step of the module.
    ///
    /// # Args
    /// * `input` - The input for one step. A tensor with shape [NUM_INPUT_FEATURES]
    ///
    /// # Returns
    /// The output tensor. Has shape [NUM_OUT_FEATURES]
    fn step(&mut self, input: &Tensor) -> Tensor;

    /// Reset the inner state for the start of a new sequence.
    ///
    /// It is not necessary to call this at the start of the first sequence.
    fn reset(&mut self);
}

/// Sequence module with stateful iteration
pub trait StatefulIterSeqModule: SequenceModule + StatefulIterativeModule {}
impl<T: SequenceModule + StatefulIterativeModule> StatefulIterSeqModule for T {}
