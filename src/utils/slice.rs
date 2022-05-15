//! Slice utilities
use super::sequence::{Sequence, TakePrefx};
use std::hash::{Hash, Hasher};
use std::iter::Chain;
use std::{fmt, slice};

/// A slice split into at most two discontiguous parts.
///
/// Acts like a single slice consisting of the concatenation of `front` and `back`.
pub struct SplitSlice<'a, T> {
    pub front: &'a [T],
    pub back: &'a [T],
}

impl<'a, T: fmt::Debug> fmt::Debug for SplitSlice<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.front, f)?;
        write!(f, " + ")?;
        fmt::Debug::fmt(&self.back, f)
    }
}

// Not derived because don't need `T: Clone`
impl<'a, T> Clone for SplitSlice<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            front: self.front,
            back: self.back,
        }
    }
}

// Not derived because don't need `T: Copy`
impl<'a, T> Copy for SplitSlice<'a, T> {}

impl<'a, T> From<&'a [T]> for SplitSlice<'a, T> {
    #[inline]
    fn from(slice: &'a [T]) -> Self {
        Self {
            front: slice,
            back: &[],
        }
    }
}

impl<'a, T> From<(&'a [T], &'a [T])> for SplitSlice<'a, T> {
    #[inline]
    fn from((front, back): (&'a [T], &'a [T])) -> Self {
        Self { front, back }
    }
}

impl<'a, T, U> PartialEq<[U]> for SplitSlice<'a, T>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, rhs: &[U]) -> bool {
        self.front.eq(&rhs[..self.front.len()]) && self.back.eq(&rhs[self.front.len()..])
    }
}

impl<'a, T, U> PartialEq<&'a [U]> for SplitSlice<'a, T>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, rhs: &&'a [U]) -> bool {
        self.eq(*rhs)
    }
}

impl<'a, 'b, T, U> PartialEq<SplitSlice<'b, U>> for SplitSlice<'a, T>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, rhs: &SplitSlice<'b, U>) -> bool {
        let self_split = self.front.len();
        let rhs_split = rhs.front.len();
        if self_split <= rhs_split {
            let (rhs_front, rhs_mid) = rhs.front.split_at(self_split);
            let (self_mid, self_back) = self.back.split_at(rhs_split - self_split);
            self.front == rhs_front && self_mid == rhs_mid && self_back == rhs.back
        } else {
            let (self_front, self_mid) = self.front.split_at(rhs_split);
            let (rhs_mid, rhs_back) = rhs.back.split_at(self_split - rhs_split);
            self_front == rhs.front && self_mid == rhs_mid && self.back == rhs_back
        }
    }
}

impl<'a, T: Eq> Eq for SplitSlice<'a, T> {}

impl<'a, T> SplitSlice<'a, T> {
    /// Return an iterator over the slice.
    #[inline]
    pub fn iter(self) -> SplitSliceIter<'a, T> {
        self.into_iter()
    }

    /// Divides the split slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` and the second will contain all indices
    /// from `[mid, len)`.
    ///
    /// # Panics
    /// Panics if `mid > len`.
    #[inline]
    #[must_use]
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        if let Some(back_split_at) = mid.checked_sub(self.front.len()) {
            let (first_back, second_back) = self.back.split_at(back_split_at);
            (
                Self {
                    front: self.front,
                    back: first_back,
                },
                second_back.into(),
            )
        } else {
            let (first_front, second_front) = self.front.split_at(mid);
            (
                first_front.into(),
                Self {
                    front: second_front,
                    back: self.back,
                },
            )
        }
    }
}

pub type SplitSliceIter<'a, T> = Chain<slice::Iter<'a, T>, slice::Iter<'a, T>>;

impl<'a, T> IntoIterator for SplitSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SplitSliceIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.front.iter().chain(self.back)
    }
}

impl<'a, T> Sequence for SplitSlice<'a, T> {
    type Item = &'a T;

    #[inline]
    fn len(&self) -> usize {
        self.front.len() + self.back.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.front.is_empty() && self.back.is_empty()
    }

    #[inline]
    fn get(&self, idx: usize) -> Option<Self::Item> {
        self.front
            .get(idx)
            .or_else(|| self.back.get(idx - self.front.len()))
    }
}

impl<'a, T> TakePrefx for SplitSlice<'a, T> {
    #[inline]
    fn take_prefix(&mut self, len: usize) -> Self {
        let (prefix, rest) = self.split_at(len);
        *self = rest;
        prefix
    }
}

impl<'a, T: Hash> Hash for SplitSlice<'a, T> {
    /// Matches the implementation of Hash for `[T]`
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Important to ensure that front & back are combined in the hash so that different splits
        // of the same full slice produce the same hashes.
        self.len().hash(state);
        Hash::hash_slice(self.front, state);
        Hash::hash_slice(self.back, state);
    }
}
