use super::super::Step;
use super::{BuildHistoryBuffer, HistoryBufferSteps};
use std::vec;

/// Configuration for [`SerialBuffer`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SerialBufferConfig {
    pub soft_threshold: usize,
    pub hard_threshold: usize,
}

impl Default for SerialBufferConfig {
    fn default() -> Self {
        Self {
            soft_threshold: 10_000,
            hard_threshold: 11_000,
        }
    }
}

/// Serial step history buffer.
///
/// The buffer records steps from a series of episodes one after another.
/// The buffer is ready when either
/// * the current episode is done and at least `soft_threshold` steps have been collected; or
/// * at least `hard_threshold` steps have been collected.
#[derive(Debug, Clone)]
pub struct SerialBuffer<O, A> {
    /// The buffer is ready when the current episode is done and at least `soft_threshold` steps
    /// have been collected.
    pub soft_threshold: usize,

    /// The buffer is ready when at least `hard_threshold` steps have been collected; even if the
    /// episode is not done.
    pub hard_threshold: usize,

    buffer: Vec<Step<O, A>>,
}

impl<O, A> BuildHistoryBuffer<O, A> for SerialBufferConfig {
    type HistoryBuffer = SerialBuffer<O, A>;

    fn build_history_buffer(&self) -> Self::HistoryBuffer {
        SerialBuffer {
            soft_threshold: self.soft_threshold,
            hard_threshold: self.hard_threshold,
            buffer: Vec::with_capacity(self.hard_threshold),
        }
    }
}

impl<O, A> SerialBuffer<O, A> {
    /// Push a new step into the buffer.
    ///
    /// Steps must be pushed consecutively within each episode.
    ///
    /// Returns a Boolean indicating whether the buffer is ready to be drained for a model update.
    pub fn push(&mut self, step: Step<O, A>) -> bool {
        let episode_done = step.episode_done;
        self.buffer.push(step);
        let num_steps = self.buffer.len();
        (episode_done && num_steps >= self.soft_threshold) || (num_steps >= self.hard_threshold)
    }
}

impl<'a, O: 'a, A: 'a> HistoryBufferSteps<'a, O, A> for SerialBuffer<O, A> {
    type StepIter = vec::Drain<'a, Step<O, A>>;

    fn drain_steps(&'a mut self) -> Self::StepIter {
        self.buffer.drain(..)
    }
}
