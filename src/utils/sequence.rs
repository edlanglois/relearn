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

/// Implement `Sequence` for a deref-able wrapper type generic over `T: Sequence + ?Sized`.
macro_rules! impl_wrapped_sequence {
    ($wrapper:ty) => {
        impl<T> Sequence for $wrapper
        where
            T: Sequence + ?Sized,
        {
            type Item = T::Item;
            #[inline]
            fn len(&self) -> usize {
                T::len(self)
            }
            #[inline]
            fn is_empty(&self) -> bool {
                T::is_empty(self)
            }
            #[inline]
            fn get(&self, idx: usize) -> Option<Self::Item> {
                T::get(self, idx)
            }
        }
    };
}
impl_wrapped_sequence!(&'_ T);
impl_wrapped_sequence!(Box<T>);

/// Take a prefix from the start of the sequence
pub trait TakePrefx: Sequence + Sized {
    /// Remove and return the prefix of length `len` from the start of the sequence.
    ///
    /// The returned sequence will contain all indices from `[0, len)` while `self` will contain
    /// all indices from `[len, initial_len)`
    ///
    /// # Panics
    /// Panics if `len > self.len()`.
    #[allow(clippy::return_self_not_must_use)] // Has side-effect on &mut self
    fn take_prefix(&mut self, len: usize) -> Self;
}

impl<'a, T> Sequence for &'a [T] {
    type Item = &'a T;
    #[inline]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    #[inline]
    fn is_empty(&self) -> bool {
        <[T]>::is_empty(self)
    }
    #[inline]
    fn get(&self, idx: usize) -> Option<Self::Item> {
        <[T]>::get(self, idx)
    }
}

impl<'a, T> TakePrefx for &'a [T] {
    #[inline]
    fn take_prefix(&mut self, index: usize) -> Self {
        let (prefix, rest) = self.split_at(index);
        *self = rest;
        prefix
    }
}

impl Sequence for Range<usize> {
    type Item = usize;

    #[inline]
    fn len(&self) -> usize {
        ExactSizeIterator::len(self)
    }

    #[inline]
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
