//! Step history buffer
use crate::Step;
use std::iter::FusedIterator;
use std::num::NonZeroUsize;
use std::ops::Range;
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

    /// Iterate over episode index ranges.
    pub fn episode_ranges(&self) -> EpisodeRangeIter {
        // Whether to include the final incomplete episode (if any)
        let mut last_end = None;
        if let Some(min_incomplete_len) = self.include_incomplete_episode_len {
            let last_complete_end: usize = self.episode_ends.last().cloned().unwrap_or(0);
            let num_steps = self.steps.len();
            if (num_steps - last_complete_end) >= min_incomplete_len.into() {
                last_end = Some(num_steps);
            }
        }
        EpisodeRangeIter::new(&self.episode_ends, last_end)
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

pub struct EpisodeRangeIter<'a> {
    /// Remaining episode ends
    episode_ends: &'a [usize],
    /// An optional episode "end" following the last index of `episode_ends`
    ///
    /// This is used to include a final incomplete episode in the list of episode ranges.
    last: Option<usize>,
    /// The current episode start.
    start: usize,
}

impl<'a> EpisodeRangeIter<'a> {
    pub const fn new(episode_ends: &'a [usize], last: Option<usize>) -> Self {
        Self {
            episode_ends,
            last,
            start: 0,
        }
    }

    /// Number of steps represented by the remaining episode ranges.
    pub const fn num_steps(&self) -> usize {
        let end = if let Some(last) = self.last {
            last
        } else if let Some(last) = self.episode_ends.last() {
            *last
        } else {
            self.start
        };
        end - self.start
    }
}

impl<'a> Iterator for EpisodeRangeIter<'a> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((&end, rest)) = self.episode_ends.split_first() {
            let range = self.start..end;
            self.start = end;
            self.episode_ends = rest;
            Some(range)
        } else if let Some(end) = self.last {
            let range = self.start..end;
            self.last = None;
            Some(range)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut len = self.episode_ends.len();
        if self.last.is_some() {
            len += 1;
        }
        (len, Some(len))
    }
}

impl<'a> ExactSizeIterator for EpisodeRangeIter<'a> {}
impl<'a> FusedIterator for EpisodeRangeIter<'a> {}
