//! Step history buffer
use crate::Step;
use std::iter;
use std::iter::Scan;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::slice::Iter;
use std::vec::Drain;

/// A step history buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryBuffer<O, A> {
    /// Step history; episodes stored contiguously one after another.
    steps: Vec<Step<O, A>>,
    /// End index of each episode.
    episode_ends: Vec<usize>,
    /// Include incomplete episodes with at least this length.
    ///
    /// * If `None`, incomplete episodes are never included.
    /// * It does not make sense to include a size-0 "episode"
    ///     because that an episoded ended on the last step,
    ///     so 0 is disallowed.
    include_incomplete_episode_len: Option<NonZeroUsize>,
}

impl<O, A> HistoryBuffer<O, A> {
    pub fn new(
        capacity: Option<usize>,
        include_incomplete_episode_len: Option<NonZeroUsize>,
    ) -> Self {
        let steps = match capacity {
            Some(c) => Vec::with_capacity(c),
            None => Vec::new(),
        };
        Self {
            steps,
            episode_ends: Vec::new(),
            include_incomplete_episode_len,
        }
    }
}

impl<O, A> HistoryBuffer<O, A> {
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of steps in the buffer.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Number of completed episodes in the buffer.
    pub fn num_complete_episodes(&self) -> usize {
        self.episode_ends.len()
    }

    /// Add a step to the buffer.
    pub fn push(&mut self, step: Step<O, A>) {
        let episode_done = step.episode_done;
        self.steps.push(step);
        if episode_done {
            self.episode_ends.push(self.steps.len());
        }
    }

    /// View the stored steps
    pub fn steps(&self) -> &[Step<O, A>] {
        &self.steps
    }

    /// Iterate over episode ranges.
    pub fn episode_ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        let num_steps = self.steps.len();

        // Whether to include the final incomplete episode (if any)
        let mut with_final_incomplete_episode = false;
        if let Some(min_incomplete_len) = self.include_incomplete_episode_len {
            let last_episode_end: usize = self.episode_ends.last().cloned().unwrap_or(0);
            with_final_incomplete_episode =
                (self.steps.len() - last_episode_end) >= min_incomplete_len.into();
        }

        self.episode_ends
            .iter()
            .cloned()
            .chain(iter::once(num_steps).take(if with_final_incomplete_episode { 1 } else { 0 }))
            .scan(0, move |start, end| {
                let range = *start..end;
                *start = end;
                Some(range)
            })
    }

    /// Creates draining iterators for the stored steps and episode ends.
    ///
    /// This fully resets the history buffer once both are drained or dropped.
    pub fn drain(&mut self) -> (Drain<Step<O, A>>, Drain<usize>) {
        (self.steps.drain(..), self.episode_ends.drain(..))
    }

    /// Clears the buffer, removing all stored data.
    pub fn clear(&mut self) {
        self.steps.clear();
        self.episode_ends.clear();
    }
}

