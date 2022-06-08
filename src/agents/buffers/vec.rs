use super::{HistoryDataBound, WriteExperience, WriteExperienceError, WriteExperienceIncremental};
use crate::simulation::PartialStep;
use crate::utils::iter::{Differences, SplitChunksByLength};
use std::iter::Copied;
use std::{slice, vec};

/// Simple vector history buffer. Stores steps in a vector.
///
/// The buffer records steps from a series of episodes one after another.
/// The buffer is ready when either
/// * the current episode is done and at least `soft_threshold` steps have been collected; or
/// * at least `hard_threshold` steps have been collected.
#[derive(Debug, Clone, PartialEq)]
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
    pub fn steps(&self) -> slice::Iter<PartialStep<O, A>> {
        self.steps.iter()
    }

    /// Draining iterator over all steps stored in the buffer.
    pub fn drain_steps(&mut self) -> vec::Drain<PartialStep<O, A>> {
        self.steps.drain(..)
    }

    /// Iterator over all episode slices stored in the buffer.
    #[must_use]
    pub fn episodes(&self) -> EpisodesIter<O, A> {
        SplitChunksByLength::new(
            &self.steps,
            Differences::new(self.episode_ends.iter().copied(), 0),
        )
    }
}

impl<O, A> From<Vec<PartialStep<O, A>>> for VecBuffer<O, A> {
    fn from(steps: Vec<PartialStep<O, A>>) -> Self {
        let episode_ends = steps
            .iter()
            .enumerate()
            .filter_map(|(i, step)| {
                if step.episode_done() {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .collect();
        let mut buffer = Self {
            steps,
            episode_ends,
        };
        buffer.end_experience();
        buffer
    }
}

impl<O, A> FromIterator<PartialStep<O, A>> for VecBuffer<O, A> {
    fn from_iter<I>(steps: I) -> Self
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
        let mut buffer = Self::new();
        buffer.write_experience(steps).unwrap(); // TODO: Maybe just ignore if full?
        buffer
    }
}

impl<O, A> WriteExperience<O, A> for VecBuffer<O, A> {
    fn write_experience<I>(&mut self, steps: I) -> Result<(), WriteExperienceError>
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
        let offset = self.steps.len();
        self.steps.extend(steps);
        for (i, step) in self.steps[offset..].iter().enumerate() {
            if step.episode_done() {
                self.episode_ends.push(offset + i + 1)
            }
        }
        self.end_experience();
        Ok(())
    }
}

impl<O, A> WriteExperienceIncremental<O, A> for VecBuffer<O, A> {
    fn write_step(&mut self, step: PartialStep<O, A>) -> Result<(), WriteExperienceError> {
        let episode_done = step.episode_done();
        self.steps.push(step);
        if episode_done {
            self.episode_ends.push(self.steps.len());
        }
        Ok(())
    }

    fn end_experience(&mut self) {
        if super::finalize_last_episode(&mut self.steps) {
            self.episode_ends.push(self.steps.len())
        }
    }
}

pub type EpisodesIter<'a, O, A> = SplitChunksByLength<
    &'a [PartialStep<O, A>],
    Differences<Copied<slice::Iter<'a, usize>>, usize>,
>;

#[allow(clippy::needless_pass_by_value)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::Successor::{self, Continue, Interrupt, Terminate};
    use rstest::{fixture, rstest};

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
    fn buffer() -> VecBuffer<usize, bool> {
        [
            step(0, Terminate),
            step(1, Continue(())),
            step(2, Terminate),
            step(3, Continue(())),
            step(4, Continue(())),
        ]
        .into_iter()
        .collect()
    }

    #[rstest]
    fn from_vec(buffer: VecBuffer<usize, bool>) {
        let test: VecBuffer<_, _> = vec![
            step(0, Terminate),
            step(1, Continue(())),
            step(2, Terminate),
            step(3, Continue(())),
            step(4, Continue(())),
        ]
        .into();
        assert_eq!(test, buffer);
    }

    #[rstest]
    fn write_experience(buffer: VecBuffer<usize, bool>) {
        let mut test = VecBuffer::new();
        test.write_experience([
            step(0, Terminate),
            step(1, Continue(())),
            step(2, Terminate),
            step(3, Continue(())),
            step(4, Continue(())),
        ])
        .unwrap();
        assert_eq!(test, buffer);
    }

    #[rstest]
    fn num_steps(buffer: VecBuffer<usize, bool>) {
        // The last step is dropped in finalization
        assert_eq!(buffer.num_steps(), 4);
    }

    #[rstest]
    fn num_episodes(buffer: VecBuffer<usize, bool>) {
        assert_eq!(buffer.num_episodes(), 3);
    }

    #[rstest]
    fn steps(buffer: VecBuffer<usize, bool>) {
        let mut steps_iter = buffer.steps();
        assert_eq!(steps_iter.next(), Some(&step(0, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(1, Continue(()))));
        assert_eq!(steps_iter.next(), Some(&step(2, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(3, Interrupt(4))));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_is_fused(buffer: VecBuffer<usize, bool>) {
        let mut steps_iter = buffer.steps();
        for _ in 0..4 {
            assert!(steps_iter.next().is_some());
        }
        assert!(steps_iter.next().is_none());
        assert!(steps_iter.next().is_none());
    }

    #[rstest]
    fn steps_len(buffer: VecBuffer<usize, bool>) {
        assert_eq!(buffer.steps().len(), buffer.num_steps());
    }

    #[rstest]
    fn episodes(buffer: VecBuffer<usize, bool>) {
        let mut episodes_iter = buffer.episodes();
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
    fn episodes_len(buffer: VecBuffer<usize, bool>) {
        assert_eq!(buffer.episodes().len(), buffer.num_episodes());
    }

    #[rstest]
    fn episode_len_sum(buffer: VecBuffer<usize, bool>) {
        assert_eq!(
            buffer.episodes().map(<[_]>::len).sum::<usize>(),
            buffer.num_steps()
        );
    }
}
