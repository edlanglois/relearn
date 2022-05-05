use crate::utils::sequence::TakePrefx;
use serde::{Deserialize, Serialize};
use std::iter::FusedIterator;

/// Iterator that splits a sequence into chunks based on an iterator of chunk lengths.
///
/// # Panics
/// Assumes that `seq` is large enough to produce all chunks specified by `lengths`.
/// Eventually panics if `seq.len()` is less than `lengths.sum()`.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SplitChunksByLength<S, I> {
    sequence: S,
    lengths: I,
}

impl<S, I> SplitChunksByLength<S, I> {
    pub fn new<U>(sequence: S, lengths: U) -> Self
    where
        U: IntoIterator<IntoIter = I>,
    {
        Self {
            sequence,
            lengths: lengths.into_iter(),
        }
    }
}

impl<S, I> Iterator for SplitChunksByLength<S, I>
where
    S: TakePrefx,
    I: Iterator<Item = usize>,
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sequence.take_prefix(self.lengths.next()?))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.lengths.size_hint()
    }

    fn count(self) -> usize {
        self.lengths.count()
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let (result, _) = self
            .lengths
            .fold((init, self.sequence), |(acc, mut seq), len| {
                let chunk = seq.take_prefix(len);
                (f(acc, chunk), seq)
            });
        result
    }
}

impl<S, I> ExactSizeIterator for SplitChunksByLength<S, I>
where
    S: TakePrefx,
    I: ExactSizeIterator<Item = usize>,
{
    fn len(&self) -> usize {
        self.lengths.len()
    }
}

impl<S, I> FusedIterator for SplitChunksByLength<S, I>
where
    S: TakePrefx,
    I: FusedIterator<Item = usize>,
{
}
