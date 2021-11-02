use super::super::Step;
use super::{BuildHistoryBuffer, HistoryBufferEpisodes, HistoryBufferSteps};
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
    /// One past the end index of each episode within `steps`.
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
            self.episode_ends.push(num_steps)
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

    fn steps(&'a self, include_incomplete: Option<usize>) -> Self::StepsIter {
        // Initialize end to the end of completed episodes
        let mut end: usize = self.episode_ends.last().cloned().unwrap_or(0);
        if let Some(min_incomplete_len) = include_incomplete {
            let num_steps = self.steps.len();
            // If there is an incomplete episode at the end with length at least min_incomplete_len
            // then change the end to be the last step.
            if (num_steps - end) >= min_incomplete_len {
                end = num_steps;
            }
        }
        self.steps[..end].iter()
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

#[allow(clippy::needless_pass_by_value)]
#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    /// Make a step that either continues or is terminal.
    const fn step(observation: usize, next_observation: Option<usize>) -> Step<usize, bool> {
        Step {
            observation,
            action: false,
            reward: 0.0,
            next_observation,
            episode_done: next_observation.is_none(),
        }
    }
    const STEP_NOT_DONE: Step<usize, bool> = step(0, Some(0));
    const STEP_TERMINAL: Step<usize, bool> = step(0, None);

    #[test]
    fn fill_episodic() {
        let mut buffer = SerialBufferConfig {
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
        let mut buffer = SerialBufferConfig {
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
    fn full_buffer() -> SerialBuffer<usize, bool> {
        let mut buffer = SerialBufferConfig {
            soft_threshold: 5,
            hard_threshold: 6,
        }
        .build_history_buffer();
        buffer.push(step(0, None));
        buffer.push(step(1, Some(2)));
        buffer.push(step(2, None));
        buffer.push(step(3, Some(4)));
        buffer.push(step(4, Some(5)));
        buffer
    }

    #[rstest]
    fn steps_no_incomplete(full_buffer: SerialBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps(None);
        assert_eq!(steps_iter.next(), Some(&step(0, None)));
        assert_eq!(steps_iter.next(), Some(&step(1, Some(2))));
        assert_eq!(steps_iter.next(), Some(&step(2, None)));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_incomplete_ge_5(full_buffer: SerialBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps(Some(5));
        assert_eq!(steps_iter.next(), Some(&step(0, None)));
        assert_eq!(steps_iter.next(), Some(&step(1, Some(2))));
        assert_eq!(steps_iter.next(), Some(&step(2, None)));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_incomplete_ge_1(full_buffer: SerialBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps(Some(1));
        assert_eq!(steps_iter.next(), Some(&step(0, None)));
        assert_eq!(steps_iter.next(), Some(&step(1, Some(2))));
        assert_eq!(steps_iter.next(), Some(&step(2, None)));
        assert_eq!(steps_iter.next(), Some(&step(3, Some(4))));
        assert_eq!(steps_iter.next(), Some(&step(4, Some(5))));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_is_fused(full_buffer: SerialBuffer<usize, bool>) {
        let mut steps_iter = full_buffer.steps(Some(0));
        for _ in 0..5 {
            assert!(steps_iter.next().is_some());
        }
        assert!(steps_iter.next().is_none());
        assert!(steps_iter.next().is_none());
    }

    #[rstest]
    fn steps_len(full_buffer: SerialBuffer<usize, bool>) {
        assert_eq!(full_buffer.steps(Some(0)).len(), 5);
    }

    #[rstest]
    fn steps_count(full_buffer: SerialBuffer<usize, bool>) {
        assert_eq!(full_buffer.steps(Some(0)).count(), 5);
    }

    #[rstest]
    fn episodes_no_incomplete(full_buffer: SerialBuffer<usize, bool>) {
        let mut episodes_iter = full_buffer.episodes(None);
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(0, None)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(1, Some(2)), &step(2, None)]
        );
        assert!(episodes_iter.next().is_none());
    }

    #[rstest]
    fn episodes_incomplete_ge_5(full_buffer: SerialBuffer<usize, bool>) {
        let mut episodes_iter = full_buffer.episodes(Some(5));
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(0, None)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(1, Some(2)), &step(2, None)]
        );
        assert!(episodes_iter.next().is_none());
    }

    #[rstest]
    fn episodes_incomplete_ge_1(full_buffer: SerialBuffer<usize, bool>) {
        let mut episodes_iter = full_buffer.episodes(Some(1));
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(0, None)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(1, Some(2)), &step(2, None)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(3, Some(4)), &step(4, Some(5))]
        );
        assert!(episodes_iter.next().is_none());
    }
}
