//! Borrowing utilities.
use serde::{Deserialize, Serialize, Serializer};
use std::borrow::{Cow, ToOwned};
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

/// Read-only smart-pointer that either owns or borrows the underlying data.
///
/// Like [`Cow`] but does not not require `T: ToOwned`.
#[derive(Copy, Clone, Deserialize)]
#[serde(from = "T")]
pub enum ReadOnly<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> From<T> for ReadOnly<'_, T> {
    #[inline]
    fn from(x: T) -> Self {
        Self::Owned(x)
    }
}

impl<'a, T> From<&'a T> for ReadOnly<'a, T> {
    #[inline]
    fn from(x: &'a T) -> Self {
        Self::Borrowed(x)
    }
}

impl<'a, T: ToOwned<Owned = T>> From<Cow<'a, T>> for ReadOnly<'a, T> {
    #[inline]
    fn from(cow: Cow<'a, T>) -> Self {
        match cow {
            Cow::Borrowed(x) => Self::Borrowed(x),
            Cow::Owned(x) => Self::Owned(x),
        }
    }
}

impl<'a, T: ToOwned<Owned = T>> From<ReadOnly<'a, T>> for Cow<'a, T> {
    #[inline]
    fn from(ro: ReadOnly<'a, T>) -> Self {
        match ro {
            ReadOnly::Borrowed(x) => Self::Borrowed(x),
            ReadOnly::Owned(x) => Self::Owned(x),
        }
    }
}

impl<T: Default> Default for ReadOnly<'_, T> {
    #[inline]
    fn default() -> Self {
        Self::Owned(T::default())
    }
}

impl<T> Deref for ReadOnly<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(x) => x,
            Self::Owned(x) => x,
        }
    }
}

impl<T> AsRef<T> for ReadOnly<'_, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T, U> PartialEq<ReadOnly<'_, U>> for ReadOnly<'_, T>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &ReadOnly<U>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: Eq> Eq for ReadOnly<'_, T> {}

impl<T, U> PartialOrd<ReadOnly<'_, U>> for ReadOnly<'_, T>
where
    T: PartialOrd<U>,
{
    #[inline]
    fn partial_cmp(&self, other: &ReadOnly<U>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: Ord> Ord for ReadOnly<'_, T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: fmt::Debug> fmt::Debug for ReadOnly<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display> fmt::Display for ReadOnly<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: Hash> Hash for ReadOnly<'_, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

impl<'a, T: Serialize> Serialize for ReadOnly<'a, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&**self, serializer)
    }
}
