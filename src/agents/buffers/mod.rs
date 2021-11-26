//! History buffers
mod iter;
mod serial;

use super::super::Step;
pub use serial::{SerialBuffer, SerialBufferConfig};

/// Build a history buffer.
pub trait BuildHistoryBuffer<O, A> {
    type HistoryBuffer;

    fn build_history_buffer(&self) -> Self::HistoryBuffer;
}

/// Access steps from a history buffer.
pub trait HistoryBufferSteps<'a, O: 'a, A: 'a> {
    /// An iterator of steps.
    type StepsIter: Iterator<Item = &'a Step<O, A>>;

    /// All steps with episode steps ordered contiguously.
    fn steps_(&'a self) -> Self::StepsIter;
}

pub trait StepsIter<'a, O: 'a, A: 'a>: Iterator<Item = &'a Step<O, A>> + ExactSizeIterator {}
impl<'a, O: 'a, A: 'a, T: ?Sized> StepsIter<'a, O, A> for T where
    T: Iterator<Item = &'a Step<O, A>> + ExactSizeIterator
{
}

/// Access steps from a history buffer via a boxed iterator.
pub trait HistoryBufferBoxedSteps<O, A> {
    fn steps<'a>(&'a self) -> Box<dyn StepsIter<O, A> + 'a>
    where
        O: 'a,
        A: 'a;
}

pub trait EpisodesIter<'a, O: 'a, A: 'a>:
    Iterator<Item = &'a [Step<O, A>]> + ExactSizeIterator
{
}
impl<'a, O: 'a, A: 'a, T: ?Sized> EpisodesIter<'a, O, A> for T where
    T: Iterator<Item = &'a [Step<O, A>]> + ExactSizeIterator
{
}

/// Access episodes from a history buffer
pub trait HistoryBufferEpisodes<'a, O: 'a, A: 'a> {
    /// An iterator of episodes. Each episode is a slice of steps.
    type EpisodesIter: Iterator<Item = &'a [Step<O, A>]>;

    /// All completed (or partial) episodes in the buffer.
    ///
    /// # Args
    /// * `include_partial` - Include partial episodes if they have at least this many steps.
    fn episodes_(&'a self) -> Self::EpisodesIter;
}

/// Access episodes from a history buffer via a boxed iterator.
pub trait HistoryBufferBoxedEpisodes<O, A> {
    fn episodes<'a>(&'a self) -> Box<dyn EpisodesIter<O, A> + 'a>
    where
        O: 'a,
        A: 'a;
}

/// Access collected episodes or steps.
pub trait HistoryBuffer<O, A>:
    HistoryBufferBoxedSteps<O, A> + HistoryBufferBoxedEpisodes<O, A>
{
}
impl<T, O, A> HistoryBuffer<O, A> for T where
    T: HistoryBufferBoxedSteps<O, A> + HistoryBufferBoxedEpisodes<O, A>
{
}
