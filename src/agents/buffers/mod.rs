//! History buffers
//!
//! # Design Note
//! There is no `ReadBuffer` trait because
//! 1. It would require Generic Associated Types (GAT) for returning iterators of slices with
//!    lifetimes tied to the lifetime of the buffer.
//! 2. A trait is not necessary because each agent uses a single specific buffer type.
//!    (Although a trait might be nice for organization).
mod null;
mod replay;
mod vec;

pub use null::NullBuffer;
pub use replay::ReplayBuffer;
pub use vec::VecBuffer;

use crate::envs::Successor;
use crate::simulation::{PartialStep, Step, StepsIter, TakeAlignedSteps};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;

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

/// Write threads of experience into a history buffer.
///
/// A thread of experience is sequence of simulation steps generated from the same `Actor`
/// and `Environment` in which each step either:
/// * ends its episode (`step.episode_done()`),
/// * is followed by the next step in the same episode, or
/// * is the last step of the iterator.
///
/// In particular, a step with `step.next == Successor::Continue` must not be followed by
/// a step from a different episode.
pub trait WriteExperience<O, A>: WriteExperienceIncremental<O, A> {
    /// Write a thread of experience into the buffer.
    fn write_experience<I>(&mut self, steps: I) -> Result<(), WriteExperienceError>
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
        Self: Sized,
    {
        for (i, step) in steps.into_iter().enumerate() {
            self.write_step(step).map_err(|e| match e {
                WriteExperienceError::Full { written_steps } => {
                    assert_eq!(
                        written_steps, 0,
                        "write_step `Full` has non-zero written steps"
                    );
                    WriteExperienceError::Full { written_steps: i }
                }
            })?;
        }
        self.end_experience();
        Ok(())
    }
}

/// Implement `WriteExperience<O, A>` for a deref-able generic wrapper type.
macro_rules! impl_wrapped_write_experience {
    ($wrapper:ty) => {
        impl<T, O, A> WriteExperience<O, A> for $wrapper where T: WriteExperience<O, A> + ?Sized {}
    };
}
impl_wrapped_write_experience!(&'_ mut T);
impl_wrapped_write_experience!(Box<T>);

/// Write threads of experience into a history history buffer one step at a time.
///
/// Prefer using [`WriteExperience`] if possible.
///
/// # Design Note
/// This interface requires that the user call `end_experience()` before reading from the buffer.
/// This could be enforced by the type system by making the incremental writer a separate object
/// (wrapping a `&mut SomeHistoryBuffer`) to be created for each experience thread and invoking the
/// `end_experience()` functionality in its destructor, when it would also release its exclusive
/// access to the buffer. This is not done for two reasons:
/// 1. It is excessively complicated for an interface that is rarely used.
/// 2. An important use-case of this interface is for things like [`SerialActorAgent`][1] that
///    persist the incremental writer while receiving steps one-by-one. The above change would make
///    this use-case even more complicated as it would require storing both the history buffer and
///    the incremental writer together when the writer (mutably) references the buffer.
///
/// [1]: super::SerialActorAgent
pub trait WriteExperienceIncremental<O, A> {
    /// Write a step into the the buffer.
    ///
    /// Successive steps must be from the same thread of experience unless `end_experience()` is
    /// called in between.
    fn write_step(&mut self, step: PartialStep<O, A>) -> Result<(), WriteExperienceError>;

    /// End the current thread of experience.
    ///
    /// This must be called before reading data from the buffer and when changing the thread of
    /// experience used for generating the steps passed to `write_step`.
    fn end_experience(&mut self);
}

/// Implement `WriteExperienceIncremental<O, A>` for a deref-able generic wrapper type.
macro_rules! impl_wrapped_write_experience_incremental {
    ($wrapper:ty) => {
        impl<T, O, A> WriteExperienceIncremental<O, A> for $wrapper
        where
            T: WriteExperienceIncremental<O, A> + ?Sized,
        {
            fn write_step(&mut self, step: PartialStep<O, A>) -> Result<(), WriteExperienceError> {
                T::write_step(self, step)
            }
            fn end_experience(&mut self) {
                T::end_experience(self)
            }
        }
    };
}
impl_wrapped_write_experience_incremental!(&'_ mut T);
impl_wrapped_write_experience_incremental!(Box<T>);

#[derive(Error, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WriteExperienceError {
    #[error("buffer full after writing {written_steps} steps")]
    Full { written_steps: usize },
}

/// Ensure that the last episode in a sequence of steps ends properly.
///
/// Helper function for internal use by buffer implementations.
///
/// Returns whether a new episode was created.
/// If a new episode was created then the last step was popped.
/// If no new episode was created then the last step may or may not have been popped.
fn finalize_last_episode<S, O, A>(steps: &mut S) -> bool
where
    S: Stack<PartialStep<O, A>>,
{
    // If the last step ends the episode (or if there is no last episode) then we're done.
    if steps.top().map_or(true, PartialStep::episode_done) {
        return false;
    }

    // If the last step of the last episode does not end the episode
    // then drop that step and interrupt the episode at the step before it.
    // Cannot interrupt at the last step because the following observation is missing.
    let final_observation = steps.pop().unwrap().observation;
    // Now check whether the formerly-second-last step ends its episode.
    // If it does then we don't need to do anything else, the step we just popped was the only one
    // in its episode. Otherwise, interrupt the episode at this new last step.
    if let Some(step) = steps.top_mut() {
        if !step.episode_done() {
            step.next = Successor::Interrupt(final_observation);
            return true;
        }
    }

    false
}

/// A stack data structure.
trait Stack<T> {
    fn push(&mut self, value: T);
    fn pop(&mut self) -> Option<T>;
    fn top(&self) -> Option<&T>;
    fn top_mut(&mut self) -> Option<&mut T>;
}

impl<T> Stack<T> for Vec<T> {
    fn push(&mut self, value: T) {
        Self::push(self, value)
    }
    fn pop(&mut self) -> Option<T> {
        Self::pop(self)
    }
    fn top(&self) -> Option<&T> {
        self.last()
    }
    fn top_mut(&mut self) -> Option<&mut T> {
        self.last_mut()
    }
}

impl<T> Stack<T> for VecDeque<T> {
    fn push(&mut self, value: T) {
        self.push_back(value)
    }
    fn pop(&mut self) -> Option<T> {
        self.pop_back()
    }
    fn top(&self) -> Option<&T> {
        self.back()
    }
    fn top_mut(&mut self) -> Option<&mut T> {
        self.back_mut()
    }
}
