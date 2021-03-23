use super::{FiniteSpace, Space};

use crate::logging::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
use std::any;
use std::fmt;
use std::marker::PhantomData;

/// An indexed set of finitely many possiblities.
pub trait Indexed {
    /// The number of possible values this type can represent.
    const SIZE: usize;

    /// Convert into an index.
    fn as_index(&self) -> usize;

    /// Create from an index.
    fn from_index(index: usize) -> Option<Self>
    where
        Self: Sized;
}

/// A space defined over an indexed type.
#[derive(Default)]
pub struct IndexedTypeSpace<T: Indexed> {
    element_type: PhantomData<T>,
}

impl<T: Indexed> fmt::Debug for IndexedTypeSpace<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IndexedTypeSpace<{}>", any::type_name::<T>())
    }
}

impl<T: Indexed> Space for IndexedTypeSpace<T> {
    type Element = T;

    fn contains(&self, _element: &Self::Element) -> bool {
        true
    }

    fn as_loggable(&self, element: &Self::Element) -> Loggable {
        Loggable::IndexSample {
            value: T::as_index(element),
            size: T::SIZE,
        }
    }
}

impl<T: Indexed> Distribution<T> for IndexedTypeSpace<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        T::from_index(rng.gen_range(0..T::SIZE)).unwrap()
    }
}

impl<T: Indexed> FiniteSpace for IndexedTypeSpace<T> {
    fn size(&self) -> usize {
        T::SIZE
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        T::as_index(element)
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        T::from_index(index)
    }
}
