//! History buffers
use super::super::Step;

/// A step history buffer.
pub trait HistoryBuffer<O, A> {
    type StepIter: Iterator<Item = Step<O, A>>;

    /// Push a new step into the buffer.
    ///
    /// Steps must be pushed consecutively within each episode.
    fn push(&mut self, step: Step<O, A>) -> bool;
    /// Drain the collected step history into an iterator of steps.
    fn drain_steps(&self) -> Self::StepIter;
}
