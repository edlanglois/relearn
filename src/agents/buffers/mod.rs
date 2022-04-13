//! History buffers
mod null;
mod simple;

pub use null::NullBuffer;
pub use simple::SimpleBuffer;

use crate::simulation::PartialStep;

/// Lower bound on the amount of data required by a buffer to ready itself.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferCapacityBound {
    /// Minimum number of steps for the buffer to store.
    pub min_steps: usize,

    /// Minimum number of episodes for the buffer to store.
    pub min_episodes: usize,

    /// Minimum length of an incomplete episode at which the buffer may indicate ready.
    ///
    /// Buffers will generally try to indicate `ready` only at episode boundaries.
    /// Setting this allows the buffer to ready itself on environments with unbounded or infinitely
    /// long episodes.
    pub min_incomplete_episode_len: Option<usize>,
}

impl BufferCapacityBound {
    /// Minimal lower bound; accept empty buffers
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            min_steps: 0,
            min_episodes: 0,
            min_incomplete_episode_len: Some(0),
        }
    }

    /// Set `min_steps` to the maximum of the current value and the given value.
    #[must_use]
    pub fn with_steps_at_least(mut self, min_steps: usize) -> Self {
        self.min_steps = self.min_steps.max(min_steps);
        self
    }

    /// Set `min_episodes` to the maximum of the current value and the given value.
    #[must_use]
    pub fn with_episodes_at_least(mut self, min_episodes: usize) -> Self {
        self.min_episodes = self.min_episodes.max(min_episodes);
        self
    }

    /// Set `min_incomplete_episode_len` to the maximum of the current value and the given value.
    #[must_use]
    pub fn with_incomplete_len_at_least(mut self, min_incomplete_episode_len: usize) -> Self {
        self.min_incomplete_episode_len = Some(
            self.min_incomplete_episode_len
                .map_or(min_incomplete_episode_len, |len| {
                    len.max(min_incomplete_episode_len)
                }),
        );
        self
    }

    /// The maximum of two bounds (field-by-field)
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self {
            min_steps: self.min_steps.max(other.min_steps),
            min_episodes: self.min_episodes.max(other.min_episodes),
            min_incomplete_episode_len: self
                .min_incomplete_episode_len
                .zip(other.min_incomplete_episode_len)
                .map(|(a, b)| a.max(b)),
        }
    }

    /// Divide the capacity into a bound of `1 / n` the size, rounding up.
    ///
    /// `n` buffers meeting the smaller bound will collectively meet the larger bound.
    #[must_use]
    pub const fn divide(self, n: usize) -> Self {
        Self {
            min_steps: div_ceil(self.min_steps, n),
            min_episodes: div_ceil(self.min_episodes, n),
            min_incomplete_episode_len: self.min_incomplete_episode_len,
        }
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

/// Add data to a history buffer.
pub trait WriteHistoryBuffer<O, A> {
    /// Insert a step into the buffer and return whether the buffer is ready for use.
    fn push(&mut self, step: PartialStep<O, A>) -> bool;

    /// Extend the buffer with steps from an iterator, stopping once ready for use.
    ///
    /// Returns whether the buffer is ready.
    fn extend_until_ready<I>(&mut self, steps: I) -> bool
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
        Self: Sized,
    {
        for step in steps {
            if self.push(step) {
                return true;
            }
        }
        false
    }

    /// Clear the buffer, removing all values.
    fn clear(&mut self);
}

/// Implement `WriteHistoryBuffer<O, A>` for a deref-able generic wrapper type.
macro_rules! impl_wrapped_write_history_buffer {
    ($wrapper:ty) => {
        impl<T, O, A> WriteHistoryBuffer<O, A> for $wrapper
        where
            T: WriteHistoryBuffer<O, A> + ?Sized,
        {
            fn push(&mut self, step: PartialStep<O, A>) -> bool {
                T::push(self, step)
            }

            fn clear(&mut self) {
                T::clear(self)
            }
        }
    };
}
impl_wrapped_write_history_buffer!(&'_ mut T);
impl_wrapped_write_history_buffer!(Box<T>);
