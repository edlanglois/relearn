//! History buffers
mod serial;

use super::super::Step;
pub use serial::{SerialBuffer, SerialBufferConfig};

/// Build a [`HistoryBuffer`].
pub trait BuildHistoryBuffer<O, A> {
    type HistoryBuffer: HistoryBuffer<O, A>;

    fn build_history_buffer(&self) -> Self::HistoryBuffer;
}

/// A step history buffer.
pub trait HistoryBuffer<O, A>: for<'a> HistoryBufferSteps<'a, O, A> {
    /// Push a new step into the buffer.
    ///
    /// Steps must be pushed consecutively within each episode.
    ///
    /// Returns a Boolean indicating whether the buffer is ready to be drained for a model update.
    fn push(&mut self, step: Step<O, A>) -> bool;
}

/// History buffer steps interface.
///
/// This takes a lifetime to work around the fact that associated types cannot yet have lifetimes.
///
/// TODO: Fix once generic associated types are stabilized.
pub trait HistoryBufferSteps<'a, O, A> {
    type StepIter: Iterator<Item = Step<O, A>>;

    /// Drain the collected step history into an iterator of steps.
    ///
    /// It is allowed to drain the buffer at any time, but the amount of available data may not
    /// meet the buffer's criteria unless the last call to `push` returned true.
    fn drain_steps(&'a mut self) -> Self::StepIter;
}
