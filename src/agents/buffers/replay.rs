use super::{WriteExperience, WriteExperienceError, WriteExperienceIncremental};
use crate::feedback::Reward;
use crate::simulation::PartialStep;
use crate::utils::iter::{Differences, SplitChunksByLength};
use crate::utils::sequence::Sequence;
use crate::utils::slice::SplitSlice;
use std::collections::{vec_deque, VecDeque};
use std::iter::{Copied, Map};

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayBuffer<O, A, F = Reward> {
    /// A circular buffer of steps.
    ///
    /// Care is taken to prevent the queue from growing beyond its initial capacity.
    steps: VecDeque<PartialStep<O, A, F>>,
    /// One past the end index of each episode in terms of `total_step_count`.
    ///
    /// Subtracting `index_offset` gives an index into the `steps` queue.
    episode_ends: VecDeque<u64>,
    /// Index of the first step of the first stored episode in terms of `total_step_count`.
    index_offset: u64,
    /// Total number of steps collected by this buffer over its lifetime.
    ///
    /// Excludes steps dropped for being the last of an incomplete episode.
    /// Includes steps from old episodes that are dropped to make room in the buffer.
    total_step_count: u64,
}

impl<O, A, F> ReplayBuffer<O, A, F> {
    /// Create a new `ReplayBuffer` with space for at least `capacity` steps.
    ///
    /// The buffer will not grow beyond its initial capacity (although this capacity might be
    /// larger than the value given here).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            steps: VecDeque::with_capacity(capacity),
            episode_ends: VecDeque::new(),
            index_offset: 0,
            total_step_count: 0,
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
        self.episode_ends.len()
    }

    /// Iterator over all steps stored in the buffer.
    #[must_use]
    pub fn steps(&self) -> vec_deque::Iter<PartialStep<O, A, F>> {
        self.steps.iter()
    }

    /// View all episodes stored in the buffer.
    #[must_use]
    pub fn episodes(&self) -> Episodes<O, A, F> {
        Episodes {
            steps: self.steps.as_slices().into(),
            episode_ends: &self.episode_ends,
            index_offset: self.index_offset,
        }
    }

    /// Total number of steps consumed by the buffer over its lifetime.
    ///
    /// Excludes any steps dropped for being the last of an incomplete episode
    /// (at most one per experience collection).
    #[must_use]
    pub const fn total_step_count(&self) -> u64 {
        self.total_step_count
    }
}

impl<O, A, F> WriteExperienceIncremental<O, A, F> for ReplayBuffer<O, A, F> {
    fn write_step(&mut self, step: PartialStep<O, A, F>) -> Result<(), WriteExperienceError> {
        if self.steps.len() == self.steps.capacity() {
            // Steps buffer is full, drop the oldest episode to free up space.
            let ep_end = self
                .episode_ends
                .pop_front()
                .ok_or(WriteExperienceError::Full { written_steps: 0 })?;
            assert!(
                ep_end > self.index_offset,
                "episodes always have at least 1 step"
            );
            // (ep_end - offset) <= self.steps.len which fits in usize
            #[allow(clippy::cast_possible_truncation)]
            let ep_len = (ep_end - self.index_offset) as usize;
            self.steps.drain(0..ep_len);
            self.index_offset = ep_end;
        }

        let episode_done = step.episode_done();
        self.steps.push_back(step);
        self.total_step_count += 1;
        if episode_done {
            self.episode_ends.push_back(self.total_step_count);
        }
        Ok(())
    }

    fn end_experience(&mut self) {
        if super::finalize_last_episode(&mut self.steps) {
            self.total_step_count -= 1; // The last step was dropped.
            self.episode_ends.push_back(self.total_step_count);
            assert_eq!(
                self.total_step_count,
                self.index_offset + self.steps.len() as u64
            );
        }
    }
}

impl<O, A, F> WriteExperience<O, A, F> for ReplayBuffer<O, A, F> {}

/// [`Sequence`] of episodes from a [`ReplayBuffer`].
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Episodes<'a, O, A, F> {
    steps: SplitSlice<'a, PartialStep<O, A, F>>,
    episode_ends: &'a VecDeque<u64>,
    index_offset: u64,
}

impl<'a, O, A, F> Sequence for Episodes<'a, O, A, F> {
    type Item = SplitSlice<'a, PartialStep<O, A, F>>;

    #[inline]
    fn len(&self) -> usize {
        self.episode_ends.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.episode_ends.is_empty()
    }

    // (ep_end - offset) <= self.steps.len which fits in usize
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    fn get(&self, idx: usize) -> Option<Self::Item> {
        // The difference between an index and self.index_offset should always fit in a usize
        // because that value indexes into steps.
        let end = (self.episode_ends.get(idx)? - self.index_offset) as usize;
        let start = if idx == 0 {
            0
        } else {
            (self.episode_ends.get(idx - 1).unwrap() - self.index_offset) as usize
        };
        assert!(end >= start);
        Some(self.steps.split_at(start).1.split_at(end - start).0)
    }
}

impl<'a, O, A, F> IntoIterator for Episodes<'a, O, A, F> {
    type IntoIter = EpisodesIter<'a, O, A, F>;
    type Item = SplitSlice<'a, PartialStep<O, A, F>>;
    fn into_iter(self) -> Self::IntoIter {
        SplitChunksByLength::new(
            self.steps,
            Differences::new(self.episode_ends.iter().copied(), self.index_offset)
                .map(u64_as_usize as _),
        )
    }
}

pub type EpisodesIter<'a, O, A, F> = SplitChunksByLength<
    SplitSlice<'a, PartialStep<O, A, F>>,
    Map<Differences<Copied<vec_deque::Iter<'a, u64>>, u64>, fn(u64) -> usize>,
>;

// used for (ep_end - offset) <= self.steps.len which fits in usize
#[allow(clippy::cast_possible_truncation)]
#[inline]
const fn u64_as_usize(x: u64) -> usize {
    x as _
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
            feedback: Reward(0.0),
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
        assert!(buffer.episodes().into_iter().eq([&ep1 as &[_]].into_iter()));

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
            .into_iter()
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
            .into_iter()
            .eq([&ep2_finalized as &[_], &ep3 as &[_]].into_iter()));

        // Two more episodes, should fit.
        let ep45 = [step(9, Terminate), step(10, Terminate)];
        buffer.write_experience(ep45).unwrap();
        assert_eq!(buffer.num_steps(), 7);
        assert_eq!(buffer.num_episodes(), 4);
        assert!(buffer
            .steps()
            .eq(ep2_finalized.iter().chain(&ep3).chain(&ep45)));
        assert!(buffer.episodes().into_iter().eq([
            &ep2_finalized as &[_],
            &ep3 as &[_],
            &ep45[..1] as &[_],
            &ep45[1..] as &[_]
        ]
        .into_iter()));
    }

    /// Check that episode random access works.
    #[test]
    fn get_episode() {
        let mut buffer = ReplayBuffer::with_capacity(7);
        // The actual capacity is an implementation detail of VecDeque.
        // Rework this unit test if the capacity is an unexpected value.
        assert_eq!(
            buffer.capacity(),
            7,
            "Implementation detail; rework test if this fails"
        );

        let data = [
            // Ep1 (dropped for space)
            step(0, Continue(())),
            step(1, Continue(())),
            step(2, Terminate),
            // Ep2
            step(3, Continue(())),
            step(4, Interrupt(5)),
            // Ep3
            step(6, Continue(())),
            step(7, Continue(())),
            step(8, Terminate),
            // Ep4
            step(9, Terminate),
            // Ep5
            step(10, Terminate),
        ];
        buffer.write_experience(data).unwrap();

        let episodes = buffer.episodes();
        assert_eq!(episodes.get(1).unwrap(), /* ep3 */ &data[5..8]);
        assert_eq!(episodes.get(3).unwrap(), /* ep5 */ &data[9..10]);
        assert!(episodes.get(4).is_none());
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
