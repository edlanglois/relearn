//! History buffers
mod serial;

use super::super::Step;
use rand::Rng;
pub use serial::{SerialBuffer, SerialBufferConfig};

/// Build a [`HistoryBuffer`].
pub trait BuildHistoryBuffer<O, A> {
    type HistoryBuffer;

    fn build_history_buffer(&self) -> Self::HistoryBuffer;
}

/// Access steps from a history buffer.
pub trait HistoryBufferSteps<'a, O: 'a, A: 'a> {
    type StepsIter: Iterator<Item = &'a Step<O, A>>;

    /// All steps with episode steps ordered contiguously.
    fn steps(&'a self) -> Self::StepsIter;
}

/// Access episodes from a history buffer
pub trait HistoryBufferEpisodes<'a, O: 'a, A: 'a>
where
    <Self::EpisodesIter as Iterator>::Item: IntoIterator<Item = &'a Step<O, A>>,
{
    /// An iterator of iterator of steps. Each inner iterator represents the steps of an episode.
    type EpisodesIter: Iterator;

    /// All completed episodes in the buffer.
    ///
    /// # Args
    /// * `include_incomplete` - Include incomplete episodes if they have at least this many steps.
    fn episodes(&'a self, include_incomplete: Option<usize>) -> Self::EpisodesIter;
}

/// Access episodes from a history buffer in a random order.
pub trait HistoryBufferShuffledEpisodes<'a, O: 'a, A: 'a>
where
    <Self::ShuffledEpisodesIter as Iterator>::Item: IntoIterator<Item = &'a Step<O, A>>,
{
    /// An iterator of iterator of steps. Each inner iterator represents the steps of an episode.
    type ShuffledEpisodesIter: Iterator;

    /// All episodes in the buffer; in a randomly shuffled order.
    ///
    /// # Args
    /// * `rng` - Random number generator for creating the shuffle.
    /// * `include_incomplete` - Include incomplete episodes if they have at least this many steps.
    fn shuffled_episodes<R: Rng + ?Sized>(
        &'a self,
        rng: &mut R,
        include_incomplete: Option<usize>,
    ) -> Self::ShuffledEpisodesIter;
}

/// Access history buffer data
pub trait HistoryBufferData<O: 'static, A: 'static>:
    for<'a> HistoryBufferSteps<'a, O, A> + for<'a> HistoryBufferEpisodes<'a, O, A>
// + for<'a> HistoryBufferShuffledEpisodes<'a, O, A>
{
}

impl<T, O, A> HistoryBufferData<O, A> for T
where
    O: 'static,
    A: 'static,
    T: for<'a> HistoryBufferSteps<'a, O, A> + for<'a> HistoryBufferEpisodes<'a, O, A>,
{
}
