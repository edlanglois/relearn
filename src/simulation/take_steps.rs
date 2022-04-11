use super::{PartialStep, Simulation};
use crate::envs::Successor;
use std::iter::FusedIterator;

/// An iterator over the first at least `n` steps of `steps`. Attempts to finish the last episode.
///
/// If the final step is [`Successor::Continue`] then it is changed to [`Successor::Interrupt`].
#[derive(Debug, Default, Clone)]
pub struct TakeAlignedSteps<I> {
    steps: I,
    /// Maximum number of steps to take.
    max_steps: usize,
    /// Slack steps. Stop taking steps if an episode ends at or after `max_steps - slack_steps`.
    slack_steps: usize,
    /// Whether the most recent episode has ended.
    episode_done: bool,
}

impl<I> TakeAlignedSteps<I> {
    pub const fn new(steps: I, min_steps: usize, slack_steps: usize) -> Self {
        Self {
            steps,
            max_steps: min_steps + slack_steps,
            slack_steps,
            episode_done: true,
        }
    }
}

impl<I> Simulation for TakeAlignedSteps<I>
where
    I: Simulation,
{
    type Observation = I::Observation;
    type Action = I::Action;
    type Environment = I::Environment;
    type Actor = I::Actor;
    type Logger = I::Logger;

    #[inline]
    fn env(&self) -> &Self::Environment {
        self.steps.env()
    }
    #[inline]
    fn env_mut(&mut self) -> &mut Self::Environment {
        self.steps.env_mut()
    }
    #[inline]
    fn actor(&self) -> &Self::Actor {
        self.steps.actor()
    }
    #[inline]
    fn actor_mut(&mut self) -> &mut Self::Actor {
        self.steps.actor_mut()
    }
    #[inline]
    fn logger(&self) -> &Self::Logger {
        self.steps.logger()
    }
    #[inline]
    fn logger_mut(&mut self) -> &mut Self::Logger {
        self.steps.logger_mut()
    }
}

impl<I, O, A> Iterator for TakeAlignedSteps<I>
where
    I: Iterator<Item = PartialStep<O, A>>,
{
    type Item = PartialStep<O, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.max_steps == 0 || (self.max_steps <= self.slack_steps && self.episode_done) {
            return None;
        }
        let mut step = self.steps.next()?;
        self.max_steps -= 1;
        self.episode_done = step.episode_done();

        // If this is the last step and the episode is not done
        // then try to change step.next from Continue to Interrupt.
        if self.max_steps == 0 && !self.episode_done {
            // Which requires the following step from the inner iterator because
            // these are PartialStep, which do not store the next observation.
            if let Some(next_step) = self.steps.next() {
                step.next = Successor::Interrupt(next_step.observation);
            }
            // If the next step is None then there is nothing we can do;
            // the information about the successor has been lost.
        }
        Some(step)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (inner_min, inner_max) = self.steps.size_hint();
        let min = inner_min.min(self.max_steps.saturating_sub(self.slack_steps));
        let max = inner_max.map_or(self.max_steps, |m| m.min(self.max_steps));
        (min, Some(max))
    }
}

impl<I, O, A> FusedIterator for TakeAlignedSteps<I> where I: FusedIterator<Item = PartialStep<O, A>> {}

#[cfg(test)]
mod tests {
    use super::super::StepsIter;
    use super::*;
    use rstest::{fixture, rstest};

    const fn step<O>(observation: O, next: Successor<O, ()>) -> PartialStep<O, ()> {
        PartialStep {
            observation,
            action: (),
            reward: 0.0,
            next,
        }
    }

    const fn cont<O>() -> Successor<O, ()> {
        Successor::Continue(())
    }

    const fn term<O>() -> Successor<O, ()> {
        Successor::Terminate
    }

    const fn interrupt<O>(observation: O) -> Successor<O, ()> {
        Successor::Interrupt(observation)
    }

    type Steps = Vec<PartialStep<u8, ()>>;

    #[fixture]
    fn steps() -> Steps {
        vec![
            // Episode 1 (len 2)
            step(0, cont()),
            step(1, term()),
            // Episode 2 (len 3)
            step(10, cont()),
            step(11, cont()),
            step(12, term()),
            // Episode 3 (len 3, interrupted)
            step(20, cont()),
            step(21, cont()),
            step(23, interrupt(23)),
        ]
    }

    #[rstest]
    fn take_no_steps(steps: Steps) {
        assert_eq!(
            steps
                .into_iter()
                .take_aligned_steps(0, 2)
                .collect::<Vec<_>>(),
            []
        );
    }

    #[rstest]
    #[allow(clippy::needless_pass_by_value)]
    fn take_all_steps(steps: Steps) {
        assert_eq!(
            steps
                .iter()
                .copied()
                .take_aligned_steps(100, 2)
                .collect::<Vec<_>>(),
            steps
        );
    }

    #[rstest]
    #[allow(clippy::needless_pass_by_value)]
    fn take_aligned_no_slack(steps: Steps) {
        assert_eq!(
            steps
                .iter()
                .copied()
                .take_aligned_steps(5, 0)
                .collect::<Vec<_>>(),
            steps[..5]
        );
    }

    #[rstest]
    #[allow(clippy::needless_pass_by_value)]
    fn take_aligned_slack(steps: Steps) {
        assert_eq!(
            steps
                .iter()
                .copied()
                .take_aligned_steps(5, 2)
                .collect::<Vec<_>>(),
            steps[..5]
        );
    }

    #[rstest]
    fn take_unaligned_no_slack(steps: Steps) {
        assert_eq!(
            steps
                .into_iter()
                .take_aligned_steps(3, 0)
                .collect::<Vec<_>>(),
            vec![step(0, cont()), step(1, term()), step(10, interrupt(11)),]
        );
    }

    #[rstest]
    fn take_unaligned_slack(steps: Steps) {
        assert_eq!(
            steps
                .iter()
                .copied()
                .take_aligned_steps(3, 2)
                .collect::<Vec<_>>(),
            steps[..5]
        );
    }
}
