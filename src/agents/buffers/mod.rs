//! History buffers
mod null;
mod vec;

pub use null::NullBuffer;
pub use vec::{VecBuffer, VecBufferEpisodes};

use crate::simulation::{PartialStep, Step, StepsIter, TakeAlignedSteps};
use serde::{Deserialize, Serialize};

/// Lower bound on an amount of actor-environment simulation steps.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HistoryDataBound {
    /// Minimum number of steps.
    pub min_steps: usize,

    /// Min number of slack steps in excess of `min_steps` when waiting for the episode to end.
    pub slack_steps: usize,
}

impl HistoryDataBound {
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            min_steps: 0,
            slack_steps: 0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn new(min_steps: usize, slack_steps: usize) -> Self {
        Self {
            min_steps,
            slack_steps,
        }
    }

    /// The maximum of two bounds (maximum of each field).
    #[inline]
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self {
            min_steps: self.min_steps.max(other.min_steps),
            slack_steps: self.slack_steps.max(other.slack_steps),
        }
    }

    /// Divide the bound into one of `1 / n` the size, rounding up.
    ///
    /// `n` buffers metting the smaller bound will collectively meet the larger bound.
    #[inline]
    #[must_use]
    pub const fn divide(self, n: usize) -> Self {
        Self {
            min_steps: div_ceil(self.min_steps, n),
            slack_steps: self.slack_steps,
        }
    }

    /// Check whether a sequence of steps satisfies the bound.
    ///
    /// # Args
    /// * `num_steps` - The number of steps in the sequence.
    /// * `last`      - The last step in the sequence (or `None` if the sequence is empty).
    #[inline]
    #[must_use]
    pub fn is_satisfied<O, A, U>(&self, num_steps: usize, last: Option<&Step<O, A, U>>) -> bool {
        num_steps >= self.min_steps && last.map_or(true, Step::episode_done)
            || num_steps >= self.min_steps + self.slack_steps
    }

    /// Apply the bound to an iterator of steps, taking the required number of steps.
    #[inline]
    pub fn take<I, O, A>(self, steps: I) -> TakeAlignedSteps<I::IntoIter>
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
        steps
            .into_iter()
            .take_aligned_steps(self.min_steps, self.slack_steps)
    }
}

/// Division rounding up
const fn div_ceil(numerator: usize, denominator: usize) -> usize {
    let mut quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder > 0 {
        quotient += 1;
    }
    quotient
}

/// Add a batch of data to a history buffer.
pub trait WriteHistoryBuffer<O, A> {
    /// Insert a step into the buffer.
    fn push(&mut self, step: PartialStep<O, A>);

    /// Insert a sequence of steps into the buffer.
    fn extend<I>(&mut self, steps: I)
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
        Self: Sized,
    {
        for step in steps {
            self.push(step)
        }
    }
}

/// Implement `WriteHistoryBuffer<O, A>` for a deref-able generic wrapper type.
macro_rules! impl_wrapped_write_history_buffer {
    ($wrapper:ty) => {
        impl<T, O, A> WriteHistoryBuffer<O, A> for $wrapper
        where
            T: WriteHistoryBuffer<O, A> + ?Sized,
        {
            fn push(&mut self, step: PartialStep<O, A>) {
                T::push(self, step)
            }
        }
    };
}
impl_wrapped_write_history_buffer!(&'_ mut T);
impl_wrapped_write_history_buffer!(Box<T>);
