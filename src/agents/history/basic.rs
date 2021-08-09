use super::super::Step;
use super::{HistoryBuffer, HistoryBufferSteps};
use std::vec;

/// Configuration for [`EpisodeBuffer`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct EpisodeBufferConfig {
    pub ep_done_step_threshold: usize,
    pub step_threshold: usize,
}

impl Default for EpisodeBufferConfig {
    fn default() -> Self {
        Self {
            ep_done_step_threshold: 10_000,
            step_threshold: 11_000,
        }
    }
}

/// Episode-based step history buffer
///
/// The buffer has a two step readyness thresholds in terms of number
/// of collected steps: `ep_done_step_threshold` and `step_threshold`.
#[derive(Debug, Clone)]
pub struct EpisodeBuffer<O, A> {
    /// The buffer is ready if the most recent episode is complete and
    /// the total number of steps is at least `ep_done_step_threshold`.
    pub ep_done_step_threshold: usize,

    /// The buffer is ready if the total number of steps is at least `step_threshold`.
    /// Setting this >= `ep_done_step_threshold` means that the buffer prefers to ready itself at
    /// an episode boundary but will eventually be ready even if the episode never ends.
    pub step_threshold: usize,

    buffer: Vec<Step<O, A>>,
}

impl<'a, O, A> From<&'a EpisodeBufferConfig> for EpisodeBuffer<O, A> {
    fn from(config: &'a EpisodeBufferConfig) -> Self {
        Self {
            ep_done_step_threshold: config.ep_done_step_threshold,
            step_threshold: config.step_threshold,
            buffer: Vec::new(),
        }
    }
}

impl<O: 'static, A: 'static> HistoryBuffer<O, A> for EpisodeBuffer<O, A> {
    fn push(&mut self, step: Step<O, A>) -> bool {
        let episode_done = step.episode_done;
        self.buffer.push(step);
        let num_steps = self.buffer.len();
        (episode_done && num_steps >= self.ep_done_step_threshold)
            || (num_steps >= self.step_threshold)
    }
}

impl<'a, O: 'a, A: 'a> HistoryBufferSteps<'a, O, A> for EpisodeBuffer<O, A> {
    type StepIter = vec::Drain<'a, Step<O, A>>;

    fn drain_steps(&'a mut self) -> Self::StepIter {
        self.buffer.drain(..)
    }
}
