use super::{HistoryDataBound, WriteHistoryBuffer};
use crate::envs::Successor;
use crate::simulation::PartialStep;
use std::iter::{Copied, ExactSizeIterator, FusedIterator};
use std::ops::{Deref, DerefMut};
use std::{slice, vec};

/// Simple vector history buffer. Stores steps in a vector.
///
/// The buffer records steps from a series of episodes one after another.
/// The buffer is ready when either
/// * the current episode is done and at least `soft_threshold` steps have been collected; or
/// * at least `hard_threshold` steps have been collected.
#[derive(Debug, Clone)]
pub struct VecBuffer<O, A> {
    /// Steps from all episodes with each episode stored contiguously
    steps: Vec<PartialStep<O, A>>,
    /// One past the end index of each episode within `steps`.
    episode_ends: Vec<usize>,
}

impl<O, A> VecBuffer<O, A> {
    /// Create a new empty [`VecBuffer`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            steps: Vec::new(),
            episode_ends: Vec::new(),
        }
    }

    /// Create a new buffer with capacity for the given amount of history data.
    #[must_use]
    pub fn with_capacity_for(bound: HistoryDataBound) -> Self {
        Self {
            steps: Vec::with_capacity(bound.min_steps.saturating_add(bound.slack_steps)),
            episode_ends: Vec::new(),
        }
    }

    /// Clear all stored data
    pub fn clear(&mut self) {
        self.steps.clear();
        self.episode_ends.clear();
    }

    /// Finalize the buffer in-place so that all episodes are well-formed.
    ///
    /// Returns a [`VecBufferEpisodes`] wrapper from which episodes can be read.
    /// The return value can ignored if one only wants to access the steps.
    pub fn finalize(&mut self) -> VecBufferEpisodes<O, A> {
        // If the last step of the last episode does not end the episode
        // then drop that step and interrupt the episode at the step before it.
        // Cannot interrupt at the last step because the following observation is missing.
        if self.steps.last().map_or(false, |step| !step.episode_done()) {
            let final_observation = self.steps.pop().unwrap().observation;
            // Now check whether the formerly-second last step ends its episode.
            // If it does then we don't need to do anything else, the step we just popped was the
            // only one in its episode.
            // Otherwise, interrupt the episode at this new last step.
            if let Some(step) = self.steps.last_mut() {
                if !step.episode_done() {
                    step.next = Successor::Interrupt(final_observation);
                    self.episode_ends.push(self.steps.len());
                }
            }
        }
        VecBufferEpisodes { buffer: self }
    }

    /// The number of steps stored in the buffer.
    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Iterator over all steps stored in the buffer.
    ///
    /// The final episode may not end properly.
    /// Use `finalize().steps()` to ensure all episodes are well-formed.
    #[must_use]
    pub fn steps(&self) -> slice::Iter<PartialStep<O, A>> {
        self.steps.iter()
    }

    /// Draining iterator over all steps stored in the buffer.
    ///
    /// The final episode may not end properly.
    /// Use `finalize().steps()` to ensure all episodes are well-formed.
    pub fn drain_steps(&mut self) -> vec::Drain<PartialStep<O, A>> {
        self.steps.drain(..)
    }
}

impl<O, A> From<Vec<PartialStep<O, A>>> for VecBuffer<O, A> {
    fn from(steps: Vec<PartialStep<O, A>>) -> Self {
        let episode_ends = steps
            .iter()
            .enumerate()
            .filter_map(|(i, step)| if step.episode_done() { Some(i) } else { None })
            .collect();
        Self {
            steps,
            episode_ends,
        }
    }
}

impl<O, A> WriteHistoryBuffer<O, A> for VecBuffer<O, A> {
    fn push(&mut self, step: PartialStep<O, A>) {
        let episode_done = step.episode_done();
        self.steps.push(step);
        if episode_done {
            self.episode_ends.push(self.steps.len());
        }
    }

    fn extend<I: IntoIterator<Item = PartialStep<O, A>>>(&mut self, steps: I) {
        let offset = self.steps.len();
        self.steps.extend(steps);
        for (i, step) in self.steps[offset..].iter().enumerate() {
            if step.episode_done() {
                self.episode_ends.push(offset + i + 1)
            }
        }
    }
}

/// An interface for reading episodes from a [`VecBuffer`]
///
/// The purpose of this struct is to ensure that all episodes in the buffer are well-formed before
/// providing access. This invariant cannot be guaranteed at all times because `VecBuffer` allows
/// pushing steps one-at-a-time and the final episode is malformed whenever it ends with
/// a continuing step.
#[derive(Debug)]
pub struct VecBufferEpisodes<'a, O, A> {
    buffer: &'a mut VecBuffer<O, A>,
}

impl<O, A> Deref for VecBufferEpisodes<'_, O, A> {
    type Target = VecBuffer<O, A>;
    fn deref(&self) -> &Self::Target {
        self.buffer
    }
}

impl<O, A> DerefMut for VecBufferEpisodes<'_, O, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
    }
}

impl<O, A> AsRef<VecBuffer<O, A>> for VecBufferEpisodes<'_, O, A> {
    fn as_ref(&self) -> &VecBuffer<O, A> {
        self.buffer
    }
}

impl<O, A> AsMut<VecBuffer<O, A>> for VecBufferEpisodes<'_, O, A> {
    fn as_mut(&mut self) -> &mut VecBuffer<O, A> {
        self.buffer
    }
}

impl<'a, O, A> VecBufferEpisodes<'a, O, A> {
    #[must_use]
    pub fn num_episodes(&self) -> usize {
        self.buffer.episode_ends.len()
    }

    #[must_use]
    pub fn episodes<'b: 'a>(&'b self) -> EpisodesIter<'a, O, A> {
        SliceChunksAtIter::new(&self.buffer.steps, self.buffer.episode_ends.iter().copied())
    }

    #[must_use]
    pub fn into_episodes(self) -> EpisodesIter<'a, O, A> {
        SliceChunksAtIter::new(&self.buffer.steps, self.buffer.episode_ends.iter().copied())
    }
}

pub type EpisodesIter<'a, O, A> =
    SliceChunksAtIter<'a, PartialStep<O, A>, Copied<slice::Iter<'a, usize>>>;

/// Iterator that partitions slice into chunks based on an iterator of end indices.
///
/// Does not include any data past the last end index.
#[derive(Debug, Clone, PartialEq)]
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

#[allow(clippy::needless_pass_by_value)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::Successor::{self, Continue, Interrupt, Terminate};
    use rstest::{fixture, rstest};

    /// Make a step that either continues or is terminal.
    const fn step(observation: usize, next: Successor<usize, ()>) -> PartialStep<usize, bool> {
        PartialStep {
            observation,
            action: false,
            reward: 0.0,
            next,
        }
    }

    /// A buffer containing (in order)
    /// * an episode of length 1,
    /// * an episode of length 2, and
    /// * 2 steps of an incomplete episode.
    #[fixture]
    fn full_buffer() -> VecBuffer<usize, bool> {
        let mut buffer = VecBuffer::new();
        buffer.extend([
            step(0, Terminate),
            step(1, Continue(())),
            step(2, Terminate),
            step(3, Continue(())),
            step(4, Continue(())),
        ]);
        buffer
    }

    #[rstest]
    fn num_steps(full_buffer: VecBuffer<usize, bool>) {
        assert_eq!(full_buffer.num_steps(), 5);
    }

    #[rstest]
    fn num_steps_finalized(mut full_buffer: VecBuffer<usize, bool>) {
        assert_eq!(full_buffer.finalize().num_steps(), 4);
    }

    #[rstest]
    fn num_episodes(mut full_buffer: VecBuffer<usize, bool>) {
        assert_eq!(full_buffer.finalize().num_episodes(), 3);
    }

    #[rstest]
    fn steps(full_buffer: VecBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps();
        assert_eq!(steps_iter.next(), Some(&step(0, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(1, Continue(()))));
        assert_eq!(steps_iter.next(), Some(&step(2, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(3, Continue(()))));
        assert_eq!(steps_iter.next(), Some(&step(4, Continue(()))));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_finalized(mut full_buffer: VecBuffer<usize, bool>) {
        let full_buffer = full_buffer.finalize();
        let mut steps_iter = full_buffer.steps();
        assert_eq!(steps_iter.next(), Some(&step(0, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(1, Continue(()))));
        assert_eq!(steps_iter.next(), Some(&step(2, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(3, Interrupt(4))));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_is_fused(full_buffer: VecBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps();
        for _ in 0..5 {
            assert!(steps_iter.next().is_some());
        }
        assert!(steps_iter.next().is_none());
        assert!(steps_iter.next().is_none());
    }

    #[rstest]
    fn steps_len(full_buffer: VecBuffer<usize, bool>) {
        assert_eq!(full_buffer.steps().len(), full_buffer.num_steps());
    }

    #[rstest]
    fn episodes(mut full_buffer: VecBuffer<usize, bool>) {
        let mut episodes_iter = full_buffer.finalize().into_episodes();
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            [&step(0, Terminate)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            [&step(1, Continue(())), &step(2, Terminate)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            [&step(3, Interrupt(4))]
        );
        assert!(episodes_iter.next().is_none());
    }

    #[rstest]
    fn episodes_len(mut full_buffer: VecBuffer<usize, bool>) {
        let buffer = full_buffer.finalize();
        assert_eq!(buffer.episodes().len(), buffer.num_episodes());
    }

    #[rstest]
    fn episode_len_sum(mut full_buffer: VecBuffer<usize, bool>) {
        let buffer = full_buffer.finalize();
        assert_eq!(
            buffer.episodes().map(<[_]>::len).sum::<usize>(),
            buffer.num_steps()
        );
    }
}
