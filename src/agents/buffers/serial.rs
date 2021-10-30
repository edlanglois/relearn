use super::super::Step;
use super::{BuildHistoryBuffer, HistoryBufferData, HistoryBufferEpisodes, HistoryBufferSteps};
use std::iter::{Chain, Cloned, ExactSizeIterator, FusedIterator};
use std::{option, slice};

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

    /// Steps from all episodes with each episode stored contiguously
    steps: Vec<Step<O, A>>,
    /// The end index of each episode within `steps`.
    episode_ends: Vec<usize>,
}

impl<O, A> BuildHistoryBuffer<O, A> for SerialBufferConfig {
    type HistoryBuffer = SerialBuffer<O, A>;

    fn build_history_buffer(&self) -> Self::HistoryBuffer {
        assert!(
            self.hard_threshold >= self.soft_threshold,
            "hard_threshold must be >= soft_threshold"
        );
        SerialBuffer {
            soft_threshold: self.soft_threshold,
            hard_threshold: self.hard_threshold,
            steps: Vec::with_capacity(self.hard_threshold),
            episode_ends: Vec::new(),
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
        self.steps.push(step);
        let num_steps = self.steps.len();
        if episode_done {
            self.episode_ends.push(num_steps - 1)
        }
        (episode_done && num_steps >= self.soft_threshold) || (num_steps >= self.hard_threshold)
    }

    pub fn clear(&mut self) {
        self.steps.clear();
        self.episode_ends.clear();
    }
}

impl<'a, O: 'a, A: 'a> HistoryBufferSteps<'a, O, A> for SerialBuffer<O, A> {
    type StepsIter = slice::Iter<'a, Step<O, A>>;

    fn steps(&'a self) -> Self::StepsIter {
        self.steps.iter()
    }
}

/// Iterator that partitions a data into chunks based on an iterator of end indices.
///
/// Does not include any data past the last end index.
pub struct SliceChunksAtIter<'a, T, I> {
    data: &'a [T],
    start: usize,
    ends: I,
}

impl<'a, T, I> SliceChunksAtIter<'a, T, I> {
    pub fn new<U: IntoIterator<IntoIter = I>>(data: &'a [T], ends: U) -> Self {
        Self {
            data,
            start: 0,
            ends: ends.into_iter(),
        }
    }
}

impl<'a, T, I> Iterator for SliceChunksAtIter<'a, T, I>
where
    I: Iterator<Item = usize>,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.ends.next()?;
        let slice = &self.data[self.start..end];
        self.start = end;
        Some(slice)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ends.size_hint()
    }

    fn count(self) -> usize {
        self.ends.count()
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (result, _) = self.ends.fold((init, self.start), |(acc, start), end| {
            (f(acc, &self.data[start..end]), end)
        });
        result
    }
}

impl<'a, T, I> ExactSizeIterator for SliceChunksAtIter<'a, T, I>
where
    I: ExactSizeIterator<Item = usize>,
{
    fn len(&self) -> usize {
        self.ends.len()
    }
}

impl<'a, T, I> FusedIterator for SliceChunksAtIter<'a, T, I> where I: FusedIterator<Item = usize> {}

pub type EpisodeStepsIter<'a, O, A> = SliceChunksAtIter<
    'a,
    Step<O, A>,
    Chain<Cloned<slice::Iter<'a, usize>>, option::IntoIter<usize>>,
>;

impl<'a, O: 'a, A: 'a> HistoryBufferEpisodes<'a, O, A> for SerialBuffer<O, A> {
    type EpisodesIter = EpisodeStepsIter<'a, O, A>;

    fn episodes(&'a self, include_incomplete: Option<usize>) -> Self::EpisodesIter {
        // Check for a partial episode at the end to include
        let mut incomplete_end = None;
        if let Some(min_incomplete_len) = include_incomplete {
            let last_complete_end: usize = self.episode_ends.last().cloned().unwrap_or(0);
            let num_steps = self.steps.len();
            if (num_steps - last_complete_end) >= min_incomplete_len {
                incomplete_end = Some(num_steps)
            }
        }

        SliceChunksAtIter::new(
            &self.steps,
            self.episode_ends.iter().cloned().chain(incomplete_end),
        )
    }
}

impl<O: 'static, A: 'static> HistoryBufferData<O, A> for SerialBuffer<O, A> {}
