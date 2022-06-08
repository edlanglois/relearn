//! Simulating agent-environment interaction
mod log_steps;
mod steps;
mod summary;
mod take_episodes;
mod take_steps;
mod train;

pub use log_steps::LogSteps;
pub use steps::Steps;
pub use summary::{OnlineStepsSummary, StepsSummary};
pub use take_episodes::TakeEpisodes;
pub use take_steps::TakeAlignedSteps;
pub use train::{train_parallel, train_serial, TrainParallelConfig};

use crate::agents::Actor;
use crate::envs::{EnvStructure, Environment, StructuredEnvironment, Successor};
use crate::logging::StatsLogger;
use rand::{Rng, SeedableRng};
use std::iter::{FusedIterator, Peekable};
use std::mem;

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

    /// Whether this step is the last of an episode.
    pub const fn episode_done(&self) -> bool {
        self.next.episode_done()
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

impl<O, A> PartialStep<O, A> {
    /// Convert a partial step into a transient step using the given reference to the next step.
    #[inline]
    pub fn into_transient_with(self, next: &Self) -> TransientStep<O, A> {
        Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: self.next.map_continue(|_: ()| &next.observation),
        }
    }

    /// Try to convert into a transient step with no successor.
    ///
    /// Succeeds so long as `self.next` is not `Successor::Continue`.
    #[inline]
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn try_into_transient<'a>(self) -> Option<TransientStep<'a, O, A>> {
        Some(Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: match self.next {
                Successor::Continue(()) => return None,
                Successor::Terminate => Successor::Terminate,
                Successor::Interrupt(obs) => Successor::Interrupt(obs),
            },
        })
    }
}

/// Seed for simulation pseudo-random state.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SimSeed {
    /// Use a random seed derived from system entropy
    Random,
    /// A root seed from which both the environment and agent random states are derived.
    Root(u64),
    /// Individual seeds for the environment and agent random states.
    Individual { env: u64, agent: u64 },
}

impl SimSeed {
    /// Derive random number generators from this seed.
    fn derive_rngs<R: Rng + SeedableRng>(self) -> (R, R) {
        match self {
            Self::Random => (R::from_entropy(), R::from_entropy()),
            Self::Root(seed) => {
                let mut env_rng = R::seed_from_u64(seed);
                // For arbitrary R, R::from_rng might produce a PRNG correlated with
                // the original. Use the slower R::seed_from_u64 instead to ensure independence.
                let agent_rng = R::seed_from_u64(env_rng.gen());
                (env_rng, agent_rng)
            }
            Self::Individual { env, agent } => (R::seed_from_u64(env), R::seed_from_u64(agent)),
        }
    }
}

/// An iterator of simulation steps.
pub trait StepsIter<O, A>: Iterator<Item = PartialStep<O, A>> {
    /// Consume all steps and produce a summary of the step statistics.
    #[inline]
    fn summarize(self) -> StepsSummary
    where
        Self: Sized,
    {
        self.collect()
    }

    /// Creates an iterator that yields steps from the first `n` episodes.
    #[inline]
    fn take_episodes(self, n: usize) -> TakeEpisodes<Self>
    where
        Self: Sized,
    {
        TakeEpisodes::new(self, n)
    }

    /// Creates an iterator of at least `min_steps` and at most `min_steps + slack` steps.
    ///
    /// Ends on the first episode boundary in this interval.
    #[inline]
    fn take_aligned_steps(self, min_steps: usize, slack: usize) -> TakeAlignedSteps<Self>
    where
        Self: Sized,
    {
        TakeAlignedSteps::new(self, min_steps, slack)
    }

    /// Fold each step viewed as a [`TransientStep`] into an accumulator using a closure.
    ///
    /// This is the equivalent of [`Iterator::fold`] on an iterator of `TransientStep`
    /// except that such an iterator cannot be created without generic associated types.
    ///
    /// If the final `step.next` is `Successor::Continue` then that step is skipped since the
    /// successor observation is missing. The code in this module avoids creating such an
    /// abruptly-ending steps iterator but it is possible with the standard iterator methods like
    /// `take(n)`.
    #[inline]
    fn fold_transient<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, TransientStep<O, A>) -> B,
        Self: Sized,
    {
        fold_transient(self, init, f)
    }

    /// Call a closure on each step viewed as a [`TransientStep`].
    ///
    /// This is the equivalent of [`Iterator::for_each`] on an iterator of `TransientStep`
    /// except that such an iterator cannot be created without generic associated types.
    ///
    /// If the final `step.next` is `Successor::Continue` then that step is skipped since the
    /// successor observation is missing. This situation should not arise if using `take_episodes`
    /// or `take_aligned_steps`.
    #[inline]
    fn for_each_transient<F>(self, mut f: F)
    where
        F: FnMut(TransientStep<O, A>),
        Self: Sized,
    {
        self.fold_transient((), |_, step| f(step));
    }

    /// Map each step viewed as a [`TransientStep`].
    ///
    /// This is the equivalent of [`Iterator::map`] on an iterator of `TransientStep`
    /// except that such an iterator cannot be created without generic associated types.
    ///
    /// If the final `step.next` is `Successor::Continue` then that step is skipped since the
    /// successor observation is missing. This situation should not arise if using `take_episodes`
    /// or `take_aligned_steps`.
    #[inline]
    fn map_transient<B, F>(self, f: F) -> MapTransient<Self, F>
    where
        F: FnMut(TransientStep<O, A>) -> B,
        Self: Sized,
    {
        MapTransient::new(self, f)
    }
}

impl<T, O, A> StepsIter<O, A> for T where T: Iterator<Item = PartialStep<O, A>> {}

#[must_use]
pub struct MapTransient<I: Iterator, F> {
    iter: Peekable<I>,
    f: F,
}

impl<I: Iterator, F> MapTransient<I, F> {
    pub fn new(iter: I, f: F) -> Self {
        Self {
            iter: iter.peekable(),
            f,
        }
    }
}

impl<I, F, O, A, B> Iterator for MapTransient<I, F>
where
    I: Iterator<Item = PartialStep<O, A>>,
    F: FnMut(TransientStep<O, A>) -> B,
{
    type Item = B;

    fn next(&mut self) -> Option<Self::Item> {
        let partial_step = self.iter.next()?;

        let step = if let Some(next) = self.iter.peek() {
            partial_step.into_transient_with(next)
        } else {
            // Drop the final step if it cannot be converted into a transient step.
            partial_step.try_into_transient()?
        };
        Some((self.f)(step))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Either one shorter or the same length as self.iter
        let (min, max) = self.iter.size_hint();
        (min.saturating_sub(1), max)
    }

    fn fold<C, G>(mut self, init: C, mut g: G) -> C
    where
        G: FnMut(C, B) -> C,
    {
        fold_transient(self.iter, init, |acc, step| g(acc, (self.f)(step)))
    }
}

fn fold_transient<I, B, F, O, A>(mut iter: I, init: B, mut f: F) -> B
where
    I: Iterator<Item = PartialStep<O, A>>,
    F: FnMut(B, TransientStep<O, A>) -> B,
{
    let mut stored = match iter.next() {
        Some(step) => step,
        None => return init,
    };

    // Fold on `self.iter` with a lag of one step:
    // At the start of each call to the closure, `stored` is the previous step.
    // We replace it with the current step (`next`) then convert the previous step
    // into a transient step with the next step reference of `stored`.
    let acc = iter.fold(init, |acc, next| {
        let step = mem::replace(&mut stored, next);
        f(acc, step.into_transient_with(&stored))
    });

    // Stored now holds the last step.
    // We can include it so long as it does not expect a successor.
    if let Some(step) = stored.try_into_transient() {
        f(acc, step)
    } else {
        acc
    }
}

impl<I, F, O, A, B> FusedIterator for MapTransient<I, F>
where
    I: FusedIterator<Item = PartialStep<O, A>>,
    F: FnMut(TransientStep<O, A>) -> B,
{
}

/// An environment-actor simulation
pub trait Simulation: StepsIter<Self::Observation, Self::Action> {
    type Observation;
    type Action;
    type Environment: Environment<Observation = Self::Observation, Action = Self::Action>;
    type Actor: Actor<Self::Observation, Self::Action>;
    type Logger: StatsLogger;

    fn env(&self) -> &Self::Environment;
    fn env_mut(&mut self) -> &mut Self::Environment;
    fn actor(&self) -> &Self::Actor;
    fn actor_mut(&mut self) -> &mut Self::Actor;
    fn logger(&self) -> &Self::Logger;
    fn logger_mut(&mut self) -> &mut Self::Logger;

    #[inline]
    /// Creates an iterator that logs each step
    fn log(
        self,
    ) -> LogSteps<
        Self,
        <Self::Environment as EnvStructure>::ObservationSpace,
        <Self::Environment as EnvStructure>::ActionSpace,
    >
    where
        Self::Environment: StructuredEnvironment,
        Self: Sized,
    {
        LogSteps::new(self)
    }
}
