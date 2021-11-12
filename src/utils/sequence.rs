//! [`Sequence`] trait
use std::ops::Range;

/// An indexable collection of a finite number of items.
pub trait Sequence {
    type Item;

    /// The number of items in the sequences.
    fn len(&self) -> usize;

    /// Check if `len() == 0`
    fn is_empty(&self) -> bool;

    /// Get an item if and only if `0 <= idx < len()`.
    fn get(&self, idx: usize) -> Option<Self::Item>;
}

impl<'a, T> Sequence for &'a [T] {
    type Item = &'a T;
    #[inline(always)]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    #[inline(always)]
    fn is_empty(&self) -> bool {
        <[T]>::is_empty(self)
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> Option<Self::Item> {
        <[T]>::get(self, idx)
    }
}

impl Sequence for Range<usize> {
    type Item = usize;

    #[inline(always)]
    fn len(&self) -> usize {
        ExactSizeIterator::len(self)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    #[inline]
    fn get(&self, idx: usize) -> Option<Self::Item> {
        let value = self.start + idx;
        if value < self.end {
            Some(value)
        } else {
            None
        }
    }
}

impl<'a> Sequence for &'a Range<usize> {
    type Item = usize;

    #[inline(always)]
    fn len(&self) -> usize {
        ExactSizeIterator::len(*self)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        (*self).is_empty()
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> Option<Self::Item> {
        (*self).get(idx)
    }
}
