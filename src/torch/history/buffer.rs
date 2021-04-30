//! Step history buffer
use crate::Step;
use std::iter::Scan;
use std::ops::Range;
use std::slice::Iter;
use std::vec::Drain;

/// A step history buffer.
pub struct HistoryBuffer<O, A> {
    /// Step history; episodes stored contiguously one after another.
    steps: Vec<Step<O, A>>,
    /// End index of each episode.
    episode_ends: Vec<usize>,
}

impl<O, A> HistoryBuffer<O, A> {
    pub fn new(capacity: Option<usize>) -> Self {
        let steps = match capacity {
            Some(c) => Vec::with_capacity(c),
            None => Vec::new(),
        };
        Self {
            steps,
            episode_ends: Vec::new(),
        }
    }
}

impl<O, A> HistoryBuffer<O, A> {
    /// Number of steps in the buffer.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Number of completed episodes in the buffer.
    pub fn num_episodes(&self) -> usize {
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

    // /// View the stored episode ends.
    // pub fn episode_ends(&self) -> &[usize] {
    //     &self.episode_ends
    // }

    /// Iterate over episode ranges.
    pub fn episode_ranges<'a>(
        &'a self,
    ) -> Scan<Iter<'a, usize>, usize, fn(&mut usize, &usize) -> Option<Range<usize>>> {
        self.episode_ends.iter().scan(0, |start, end| {
            let range = *start..*end;
            *start = *end;
            Some(range)
        })
    }

    // /// Iterate over episode lengths.
    // pub fn episode_lengths<'a>(
    //     &'a self,
    // ) -> Scan<Iter<'a, usize>, usize, fn(&mut usize, &usize) -> Option<usize>> {
    //     self.episode_ends.iter().scan(0, |start, end| {
    //         let length = *end - *start;
    //         *start = *end;
    //         Some(length)
    //     })
    // }

    /// Creates draining iterators for the stored steps and episode ends.
    ///
    /// This fully resets the history buffer once both are drained or dropped.
    pub fn drain<'a>(&'a mut self) -> (Drain<'a, Step<O, A>>, Drain<'a, usize>) {
        (self.steps.drain(..), self.episode_ends.drain(..))
    }

    // /// Clears the buffer, removing all stored data.
    // pub fn clear(&mut self) {
    //     self.steps.clear();
    //     self.episode_ends.clear();
    // }
}
