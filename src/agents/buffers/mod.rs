//! History buffers
mod serial;
mod vec;

use crate::simulation::PartialStep;
pub use serial::{SerialBuffer, SerialBufferConfig};

/// Build a history buffer.
pub trait BuildHistoryBuffer<O, A> {
    type HistoryBuffer;

    fn build_history_buffer(&self) -> Self::HistoryBuffer;
}

/// Access collected episodes or steps.
pub trait HistoryBuffer<O, A> {
    /// Total number of steps
    ///
    /// Equal to `self.steps().len()` and `self.episodes().map(|e| e.len()).sum()`.
    fn num_steps(&self) -> usize;

    /// Total number of episodes (may include incomplete episodes)
    ///
    /// Equal to `self.episodes().len()`.
    fn num_episodes(&self) -> usize;

    /// All steps ordered contiguously by episode.
    fn steps<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a PartialStep<O, A>> + 'a>;

    /// Drain all steps from the buffer (ordered contiguously by episode).
    ///
    /// The buffer will be empty once the iterator is consumed or dropped.
    fn drain_steps(&mut self) -> Box<dyn ExactSizeIterator<Item = PartialStep<O, A>> + '_>;

    /// All episodes (including incomplete episodes).
    fn episodes<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a [PartialStep<O, A>]> + 'a>;
}
