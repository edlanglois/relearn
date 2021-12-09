use super::{BuildHistoryBuffer, HistoryBuffer, WriteHistoryBuffer};
use crate::simulation::PartialStep;
use crate::utils::iter::SizedChain;
use std::iter::{ExactSizeIterator, FusedIterator};

/// Configuration for [`SimpleBuffer`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SimpleBufferConfig {
    pub soft_threshold: usize,
    pub hard_threshold: usize,
}

impl Default for SimpleBufferConfig {
    fn default() -> Self {
        Self {
            soft_threshold: 10_000,
            hard_threshold: 11_000,
        }
    }
}

/// Simple history buffer. Stores and replays observed steps without additional processing.
///
/// The buffer records steps from a series of episodes one after another.
/// The buffer is ready when either
/// * the current episode is done and at least `soft_threshold` steps have been collected; or
/// * at least `hard_threshold` steps have been collected.
#[derive(Debug, Clone)]
pub struct SimpleBuffer<O, A> {
    /// The buffer is ready when the current episode is done and at least `soft_threshold` steps
    /// have been collected.
    pub soft_threshold: usize,

    /// The buffer is ready when at least `hard_threshold` steps have been collected; even if the
    /// episode is not done.
    pub hard_threshold: usize,

    /// Steps from all episodes with each episode stored contiguously
    steps: Vec<PartialStep<O, A>>,
    /// One past the end index of each episode within `steps`.
    episode_ends: Vec<usize>,
}

impl<O, A> BuildHistoryBuffer<O, A> for SimpleBufferConfig {
    type HistoryBuffer = SimpleBuffer<O, A>;

    fn build_history_buffer(&self) -> Self::HistoryBuffer {
        assert!(
            self.hard_threshold >= self.soft_threshold,
            "hard_threshold must be >= soft_threshold"
        );
        SimpleBuffer {
            soft_threshold: self.soft_threshold,
            hard_threshold: self.hard_threshold,
            steps: Vec::with_capacity(self.hard_threshold),
            episode_ends: Vec::new(),
        }
    }
}

impl<O, A> WriteHistoryBuffer<O, A> for SimpleBuffer<O, A> {
    fn push(&mut self, step: PartialStep<O, A>) -> bool {
        let episode_done = step.next.episode_done();
        self.steps.push(step);
        let num_steps = self.steps.len();
        if episode_done {
            self.episode_ends.push(num_steps)
        }
        (episode_done && num_steps >= self.soft_threshold) || (num_steps >= self.hard_threshold)
    }

    fn clear(&mut self) {
        self.steps.clear();
        self.episode_ends.clear();
    }
}

impl<O, A> HistoryBuffer<O, A> for SimpleBuffer<O, A> {
    fn num_steps(&self) -> usize {
        self.steps.len()
    }

    fn num_episodes(&self) -> usize {
        if self.steps.len() > self.episode_ends.last().cloned().unwrap_or(0) {
            self.episode_ends.len() + 1
        } else {
            self.episode_ends.len()
        }
    }

    fn steps<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a PartialStep<O, A>> + 'a> {
        Box::new(self.steps.iter())
    }

    fn drain_steps(&mut self) -> Box<dyn ExactSizeIterator<Item = PartialStep<O, A>> + '_> {
        Box::new(self.steps.drain(..))
    }

    fn episodes<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a [PartialStep<O, A>]> + 'a> {
        // Check for an incomplete episode at the end to include
        let last_complete_end: usize = self.episode_ends.last().cloned().unwrap_or(0);
        let num_steps = self.steps.len();
        let incomplete_end = if num_steps > last_complete_end {
            Some(num_steps)
        } else {
            None
        };

        let ends_iter: SizedChain<_, _> = self
            .episode_ends
            .iter()
            .cloned()
            .chain(incomplete_end)
            .into();
        Box::new(SliceChunksAtIter::new(&self.steps, ends_iter))
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

#[allow(clippy::needless_pass_by_value)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::Successor::{self, Continue, Terminate};
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
    const STEP_NOT_DONE: PartialStep<usize, bool> = step(0, Continue(()));
    const STEP_TERMINAL: PartialStep<usize, bool> = step(0, Terminate);

    #[test]
    fn fill_episodic() {
        let mut buffer = SimpleBufferConfig {
            soft_threshold: 3,
            hard_threshold: 5,
        }
        .build_history_buffer();
        assert!(!buffer.push(STEP_NOT_DONE));
        assert!(!buffer.push(STEP_TERMINAL));
        assert!(!buffer.push(STEP_NOT_DONE)); // Soft threshold but episode not done
        assert!(buffer.push(STEP_TERMINAL)); // Soft threshold and episode done
    }

    #[test]
    fn fill_non_episodic() {
        let mut buffer = SimpleBufferConfig {
            soft_threshold: 3,
            hard_threshold: 5,
        }
        .build_history_buffer();
        assert!(!buffer.push(STEP_NOT_DONE));
        assert!(!buffer.push(STEP_NOT_DONE));
        assert!(!buffer.push(STEP_NOT_DONE));
        assert!(!buffer.push(STEP_NOT_DONE));
        assert!(buffer.push(STEP_NOT_DONE)); // Hard threshold
    }

    #[fixture]
    /// A buffer containing (in order)
    /// * an episode of length 1,
    /// * an episode of length 2, and
    /// * 2 steps of an incomplete episode.
    fn full_buffer() -> SimpleBuffer<usize, bool> {
        let mut buffer = SimpleBufferConfig {
            soft_threshold: 5,
            hard_threshold: 6,
        }
        .build_history_buffer();
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
    fn num_steps(full_buffer: SimpleBuffer<usize, bool>) {
        assert_eq!(full_buffer.num_steps(), 5);
    }

    #[rstest]
    fn num_episodes(full_buffer: SimpleBuffer<usize, bool>) {
        assert_eq!(full_buffer.num_episodes(), 3);
    }

    #[rstest]
    fn steps(full_buffer: SimpleBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps();
        assert_eq!(steps_iter.next(), Some(&step(0, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(1, Continue(()))));
        assert_eq!(steps_iter.next(), Some(&step(2, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(3, Continue(()))));
        assert_eq!(steps_iter.next(), Some(&step(4, Continue(()))));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_is_fused(full_buffer: SimpleBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps();
        for _ in 0..5 {
            assert!(steps_iter.next().is_some());
        }
        assert!(steps_iter.next().is_none());
        assert!(steps_iter.next().is_none());
    }

    #[rstest]
    fn steps_len(full_buffer: SimpleBuffer<usize, bool>) {
        assert_eq!(full_buffer.steps().len(), full_buffer.num_steps());
    }

    #[rstest]
    fn episodes_incomplete_ge_1(full_buffer: SimpleBuffer<usize, bool>) {
        let mut episodes_iter = full_buffer.episodes();
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(0, Terminate)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(1, Continue(())), &step(2, Terminate)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(3, Continue(())), &step(4, Continue(()))]
        );
        assert!(episodes_iter.next().is_none());
    }

    #[rstest]
    fn episodes_len(full_buffer: SimpleBuffer<usize, bool>) {
        assert_eq!(full_buffer.episodes().len(), full_buffer.num_episodes());
    }

    #[rstest]
    fn episode_len_sum(full_buffer: SimpleBuffer<usize, bool>) {
        assert_eq!(
            full_buffer.episodes().map(|e| e.len()).sum::<usize>(),
            full_buffer.num_steps()
        );
    }
}
