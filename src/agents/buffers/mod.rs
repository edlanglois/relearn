//! History buffers
mod simple;

use crate::simulation::PartialStep;
pub use simple::{SimpleBuffer, SimpleBufferConfig};

/// Build a history buffer.
pub trait BuildHistoryBuffer<O, A> {
    type HistoryBuffer;

    fn build_history_buffer(&self) -> Self::HistoryBuffer;
}

/// Add data to a history buffer.
pub trait WriteHistoryBuffer<O, A> {
    /// Insert a step into the buffer and return whether the buffer is full.
    fn push(&mut self, step: PartialStep<O, A>) -> bool;

    /// Extend the buffer with steps from an iterator, stopping once full.
    ///
    /// Returns whether the buffer is full.
    fn extend<I>(&mut self, steps: I) -> bool
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
        for step in steps {
            if self.push(step) {
                return true;
            }
        }
        false
    }

    /// Clear the buffer, removing all values.
    fn clear(&mut self);
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
