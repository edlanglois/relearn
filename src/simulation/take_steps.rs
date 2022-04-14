use super::{PartialStep, Simulation};
use std::iter::FusedIterator;

/// An iterator over the next at most `n` steps of `steps`. Attempts to complete the last episode.
///
/// Ends on the first episode boundary between `n` and `n - slack_steps`.
/// May leave the last episode dangling (last `step.next` is `Successor::Continue`).
#[derive(Debug, Default, Clone)]
pub struct TakeAlignedSteps<I> {
    steps: I,
    /// Maximum number of steps to take.
    n: usize,
    /// Slack steps. Stop taking steps if an episode ends at or after `n - slack_steps`.
    slack_steps: usize,
}

impl<I> TakeAlignedSteps<I> {
    pub const fn new(steps: I, min_steps: usize, slack_steps: usize) -> Self {
        let n = if min_steps == 0 {
            // At an episode boundary already so no need to take any steps even if slack > 0
            0
        } else {
            min_steps + slack_steps
        };
        Self {
            steps,
            n,
            slack_steps,
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
        if self.n == 0 {
            return None;
        }

        let step = self.steps.next()?;
        self.n -= 1;
        if step.episode_done() && self.n <= self.slack_steps {
            // Ended within the slack interval. Stop here.
            self.n = 0;
        }
        Some(step)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (inner_min, inner_max) = self.steps.size_hint();
        let min = inner_min.min(self.n.saturating_sub(self.slack_steps));
        let max = inner_max.map_or(self.n, |m| m.min(self.n));
        (min, Some(max))
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut n = self.n;
        let slack_steps = self.slack_steps;
        self.take_while(move |step| {
            if n == 0 {
                return false;
            }
            n -= 1;
            if step.episode_done() && n <= slack_steps {
                n = 0;
            }
            true
        })
        .fold(init, f)
    }
}

impl<I, O, A> FusedIterator for TakeAlignedSteps<I> where I: FusedIterator<Item = PartialStep<O, A>> {}

#[cfg(test)]
#[allow(clippy::needless_pass_by_value)]
mod tests {
    use super::super::StepsIter;
    use super::*;
    use crate::envs::Successor::{self, Continue, Interrupt, Terminate};
    use rstest::{fixture, rstest};

    const fn step<O>(observation: O, next: Successor<O, ()>) -> PartialStep<O, ()> {
        PartialStep {
            observation,
            action: (),
            reward: 0.0,
            next,
        }
    }

    type Steps = Vec<PartialStep<u8, ()>>;

    #[fixture]
    fn steps() -> Steps {
        vec![
            // Episode 1 (len 2)
            step(0, Continue(())),
            step(1, Terminate),
            // Episode 2 (len 3)
            step(10, Continue(())),
            step(11, Continue(())),
            step(12, Terminate),
            // Episode 3 (len 3, interrupted)
            step(20, Continue(())),
            step(21, Continue(())),
            step(23, Interrupt(23)),
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
                .iter()
                .copied()
                .take_aligned_steps(3, 0)
                .collect::<Vec<_>>(),
            steps[..3]
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
