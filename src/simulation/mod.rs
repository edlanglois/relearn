//! Simulating agent-environment interaction
mod steps;
mod summary;
mod take_episodes;
mod train;

pub use steps::{SimSeed, SimulatorSteps};
pub use summary::{OnlineStepsSummary, StepsSummary};
pub use take_episodes::TakeEpisodes;
pub use train::{train_parallel, train_serial, TrainParallelConfig};

use crate::envs::Successor;

/// Description of an environment step.
///
/// There are a few different forms that this structure can take in terms of describing the
/// next observation when `next` is [`Successor::Continue`].
/// These are determined by the value of the third generic parameter `U`:
/// * `Step<O, A>` - `U = O` - The continuing successor observation is owned.
/// * [`TransientStep<O, A>`] - `U = &O` - The continuing successor observation is borrowed.
/// * [`PartialStep<O, A>`] - `U = ()` - The continuing successor observation is omitted.
///
/// If `next` is [`Successor::Interrupt`] then the observation is owned in all cases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step<O, A, U = O> {
    /// The initial observation.
    pub observation: O,
    /// The action taken from the initial state given the initial observation.
    pub action: A,
    /// The resulting reward.
    pub reward: f64,
    /// The next observation or outcome; how the episode progresses.
    pub next: Successor<O, U>,
}

impl<O, A, U> Step<O, A, U> {
    pub const fn new(observation: O, action: A, reward: f64, next: Successor<O, U>) -> Self {
        Self {
            observation,
            action,
            reward,
            next,
        }
    }

    pub fn into_partial(self) -> PartialStep<O, A> {
        Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: self.next.into_partial(),
        }
    }
}

/// Trait for simulation step iterators.
pub trait StepsIter<O, A>: Iterator<Item = PartialStep<O, A>> {
    /// Creates an iterator that yields steps from the first `n` episodes.
    #[inline]
    fn take_episodes(self, n: usize) -> TakeEpisodes<Self>
    where
        Self: Sized,
    {
        TakeEpisodes::new(self, n)
    }
}

impl<T, O, A> StepsIter<O, A> for T where T: Iterator<Item = PartialStep<O, A>> {}

/// Description of an environment step where the successor observation is borrowed.
pub type TransientStep<'a, O, A> = Step<O, A, &'a O>;

impl<O: Clone, A> TransientStep<'_, O, A> {
    /// Convert a transient step into an owned step by cloning any borrowed successor observation.
    #[inline]
    pub fn into_owned(self) -> Step<O, A> {
        Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: self.next.into_owned(),
        }
    }
}

/// Partial description of an environment step.
///
/// The successor state is omitted when the episode continues.
/// Using this can help avoid copying the observation.
pub type PartialStep<O, A> = Step<O, A, ()>;
