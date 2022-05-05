use std::iter::FusedIterator;
use std::mem;
use std::ops::Sub;

/// Converts a sequence of values into a sequence of differents.
///
/// Takes an initial value from which the difference to the first iterator item is calculated.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Differences<I, T> {
    iter: I,
    prev: T,
}

impl<I, T> Differences<I, T> {
    pub const fn new(iter: I, initial_value: T) -> Self {
        Self {
            iter,
            prev: initial_value,
        }
    }
}

impl<I, T> Iterator for Differences<I, T>
where
    I: Iterator<Item = T>,
    T: Sub + Copy,
{
    type Item = T::Output;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let prev = mem::replace(&mut self.prev, self.iter.next()?);
        Some(self.prev - prev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter
            .fold((init, self.prev), |(acc, prev), next| {
                (f(acc, next - prev), next)
            })
            .0
    }
}

impl<I, T> ExactSizeIterator for Differences<I, T>
where
    I: Iterator<Item = T>,
    T: Sub + Copy,
{
}
impl<I, T> FusedIterator for Differences<I, T>
where
    I: Iterator<Item = T>,
    T: Sub + Copy,
{
}
