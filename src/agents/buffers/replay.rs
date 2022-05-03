use super::{WriteExperience, WriteExperienceError, WriteExperienceIncremental};
use crate::simulation::PartialStep;
use crate::utils::iter::SizedChain;
use crate::utils::sequence::Sequence;
use std::collections::{vec_deque, VecDeque};
use std::iter::{Copied, FusedIterator};
use std::slice;

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayBuffer<O, A> {
    /// A circular buffer of steps.
    ///
    /// Care is taken to prevent the queue from growing beyond its initial capacity.
    steps: VecDeque<PartialStep<O, A>>,
    /// The length of each complete episode in `steps`.
    episode_lengths: VecDeque<usize>,
    /// The length of the current episode.
    current_episode_len: usize,
}

impl<O, A> ReplayBuffer<O, A> {
    /// Create a new `ReplayBuffer` with space for at least `capacity` steps.
    ///
    /// The buffer will not grow beyond its initial capacity (although this capacity might be
    /// larger than the value given here).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            steps: VecDeque::with_capacity(capacity),
            episode_lengths: VecDeque::new(),
            current_episode_len: 0,
        }
    }

    /// The step capacity of the buffer. Will not grow beyond this capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.steps.capacity()
    }

    /// The number of steps stored in the buffer.
    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// The number of episodes stored in the buffer.
    #[must_use]
    pub fn num_episodes(&self) -> usize {
        self.episode_lengths.len()
    }

    /// Iterator over all steps stored in the buffer.
    #[must_use]
    pub fn steps(&self) -> vec_deque::Iter<PartialStep<O, A>> {
        self.steps.iter()
    }

    /// Iterator over all episode slices stored in the buffer.
    #[must_use]
    pub fn episodes(&self) -> EpisodesIter<O, A> {
        VecDequeChunksByLength::new(&self.steps, self.episode_lengths.iter().copied())
    }
}

impl<O, A> WriteExperienceIncremental<O, A> for ReplayBuffer<O, A> {
    fn write_step(&mut self, step: PartialStep<O, A>) -> Result<(), WriteExperienceError> {
        if self.steps.len() == self.steps.capacity() {
            // Steps buffer is full, drop the oldest episode to free up space.
            let len = self
                .episode_lengths
                .pop_front()
                .ok_or(WriteExperienceError::Full { written_steps: 0 })?;
            assert!(len > 0, "episodes always have at least 1 step");
            self.steps.drain(0..len);
        }

        let episode_done = step.episode_done();
        self.steps.push_back(step);
        self.current_episode_len += 1;
        if episode_done {
            self.episode_lengths.push_back(self.current_episode_len);
            self.current_episode_len = 0;
        }
        Ok(())
    }

    fn end_experience(&mut self) {
        if super::finalize_last_episode(&mut self.steps) {
            self.episode_lengths
                .push_back(self.current_episode_len.checked_sub(1).unwrap())
        } else {
            assert!(self.current_episode_len <= 1);
        }
        self.current_episode_len = 0;
    }
}

impl<O, A> WriteExperience<O, A> for ReplayBuffer<O, A> {}

pub type EpisodesIter<'a, O, A> =
    VecDequeChunksByLength<'a, PartialStep<O, A>, Copied<vec_deque::Iter<'a, usize>>>;

/// A chunk (set of sequential elements) from a [`VecDeque`]
#[derive(Debug, Copy, Clone)]
pub enum VecDequeChunk<'a, T> {
    Slice(&'a [T]),
    Split(&'a [T], &'a [T]),
}

impl<'a, T, U> PartialEq<[U]> for VecDequeChunk<'a, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &[U]) -> bool {
        match self {
            Self::Slice(slice) => slice.eq(&rhs),
            Self::Split(front, back) => {
                front.eq(&&rhs[..front.len()]) && back.eq(&&rhs[front.len()..])
            }
        }
    }
}

impl<'a, T, U> PartialEq<&'a [U]> for VecDequeChunk<'a, T>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &&'a [U]) -> bool {
        self.eq(*rhs)
    }
}

impl<'a, T> IntoIterator for VecDequeChunk<'a, T> {
    type Item = &'a T;
    type IntoIter = VecDequeChunkIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> VecDequeChunk<'a, T> {
    fn iter(&self) -> VecDequeChunkIter<'a, T> {
        let (front, back) = match *self {
            Self::Slice(slice) => (slice, &[] as &[_]),
            Self::Split(front, back) => (front, back),
        };
        front.iter().chain(back).into()
    }
}

impl<'a, T> Sequence for VecDequeChunk<'a, T> {
    type Item = &'a T;

    fn len(&self) -> usize {
        match self {
            Self::Slice(slice) => slice.len(),
            Self::Split(front, back) => front.len() + back.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Slice(slice) => slice.is_empty(),
            Self::Split(front, back) => front.is_empty() && back.is_empty(),
        }
    }

    fn get(&self, idx: usize) -> Option<Self::Item> {
        match self {
            Self::Slice(slice) => slice.get(idx),
            Self::Split(front, back) => front.get(idx).or_else(|| back.get(idx - front.len())),
        }
    }
}

/// Iterator over elements of a [`VecDequeChunk`].
pub type VecDequeChunkIter<'a, T> = SizedChain<slice::Iter<'a, T>, slice::Iter<'a, T>>;

/// Iterator over chunks from a `VecDeque` with chunk lengths given by an iterator.
///
/// Panics if `lengths` takes more data than is available in the queue.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecDequeChunksByLength<'a, T, I> {
    front: &'a [T],
    back: Option<&'a [T]>,
    lengths: I,
}

impl<'a, T, I> VecDequeChunksByLength<'a, T, I> {
    pub fn new<U: IntoIterator<IntoIter = I>>(data: &'a VecDeque<T>, lengths: U) -> Self {
        let (front, back) = data.as_slices();
        Self {
            front,
            back: Some(back),
            lengths: lengths.into_iter(),
        }
    }

    /// Take a slice of the given size from the start of `front`.
    ///
    /// If `front` is not long enough then takes all remaining data leaving an empty slice.
    fn take_from_front(&mut self, size: usize) -> &'a [T] {
        let (chunk, rest) = self.front.split_at(size.min(self.front.len()));
        self.front = rest;
        chunk
    }
}

impl<'a, T, I> Iterator for VecDequeChunksByLength<'a, T, I>
where
    I: Iterator<Item = usize>,
{
    type Item = VecDequeChunk<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk_length = self.lengths.next()?;

        let chunk_front = self.take_from_front(chunk_length);
        if chunk_front.len() == chunk_length {
            return Some(VecDequeChunk::Slice(chunk_front));
        }

        // If the front was too small then replace with the back slice to get the rest.
        self.front = self.back.take().expect("out of data for chunk");

        let chunk_back = self.take_from_front(chunk_length - chunk_front.len());
        assert_eq!(
            chunk_front.len() + chunk_back.len(),
            chunk_length,
            "out of data for chunk"
        );
        Some(VecDequeChunk::Split(chunk_front, chunk_back))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.lengths.size_hint()
    }

    fn count(self) -> usize {
        self.lengths.count()
    }
}

impl<'a, T, I> ExactSizeIterator for VecDequeChunksByLength<'a, T, I>
where
    I: ExactSizeIterator<Item = usize>,
{
    fn len(&self) -> usize {
        self.lengths.len()
    }
}

impl<'a, T, I> FusedIterator for VecDequeChunksByLength<'a, T, I> where
    I: FusedIterator<Item = usize>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::Successor::{self, Continue, Interrupt, Terminate};
    use std::iter;

    const fn step(observation: usize, next: Successor<usize, ()>) -> PartialStep<usize, bool> {
        PartialStep {
            observation,
            action: false,
            reward: 0.0,
            next,
        }
    }

    /// Comprehensive test of [`ReplayBuffer`] happy-path
    #[test]
    fn comprehensive() {
        let mut buffer = ReplayBuffer::with_capacity(7);
        // The actual capacity is an implementation detail of VecDeque.
        // Rework this unit test if the capacity is an unexpected value.
        assert_eq!(
            buffer.capacity(),
            7,
            "Implementation detail; rework test if this fails"
        );

        // Basic initial episode
        let ep1 = [
            step(0, Continue(())),
            step(1, Continue(())),
            step(2, Terminate),
        ];
        buffer.write_experience(ep1).unwrap();
        assert_eq!(buffer.num_steps(), 3);
        assert_eq!(buffer.num_episodes(), 1);
        assert!(buffer.steps().eq(&ep1));
        assert!(buffer.episodes().eq([&ep1 as &[_]].into_iter()));

        // Second episode -- not terminated
        let ep2_raw = [
            step(3, Continue(())),
            step(4, Continue(())),
            step(5, Continue(())),
        ];
        let ep2_finalized = [step(3, Continue(())), step(4, Interrupt(5))];
        buffer.write_experience(ep2_raw).unwrap();
        assert_eq!(buffer.num_steps(), 5);
        assert_eq!(buffer.num_episodes(), 2);
        assert!(buffer.steps().eq(ep1.iter().chain(&ep2_finalized)));
        assert!(buffer
            .episodes()
            .eq([&ep1 as &[_], &ep2_finalized as &[_]].into_iter()));

        // Third episode, overflow and causes the first episode to be dropped
        let ep3 = [
            step(6, Continue(())),
            step(7, Continue(())),
            step(8, Terminate),
        ];
        buffer.write_experience(ep3).unwrap();
        assert_eq!(buffer.num_steps(), 5);
        assert_eq!(buffer.num_episodes(), 2);
        assert!(buffer.steps().eq(ep2_finalized.iter().chain(&ep3)));
        assert!(buffer
            .episodes()
            .eq([&ep2_finalized as &[_], &ep3 as &[_]].into_iter()));

        // Two more episodes, should fit.
        let ep45 = [step(9, Terminate), step(10, Terminate)];
        buffer.write_experience(ep45).unwrap();
        assert_eq!(buffer.num_steps(), 7);
        assert_eq!(buffer.num_episodes(), 4);
        assert!(buffer
            .steps()
            .eq(ep2_finalized.iter().chain(&ep3).chain(&ep45)));
        assert!(buffer.episodes().eq([
            &ep2_finalized as &[_],
            &ep3 as &[_],
            &ep45[..1] as &[_],
            &ep45[1..] as &[_]
        ]
        .into_iter()));
    }

    /// Check that a writing a too-large episode fails.
    #[test]
    fn episode_too_large() {
        let mut buffer = ReplayBuffer::with_capacity(7);
        // 100 is hopefully more than the actual capacity of `buffer`.
        let result = buffer.write_experience(iter::repeat(step(0, Continue(()))).take(100));
        assert!(matches!(
            result,
            Err(WriteExperienceError::Full { written_steps: _ })
        ));
    }
}
