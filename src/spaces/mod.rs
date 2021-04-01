//! Space definitions
mod finite;
mod index;
mod indexed_type;
mod singleton;
#[cfg(test)]
pub mod testing;

pub use finite::FiniteSpace;
pub use index::IndexSpace;
pub use indexed_type::{Indexed, IndexedTypeSpace};
pub use singleton::SingletonSpace;

use crate::logging::Loggable;
use rand::distributions::Distribution;

/// A mathematical space
pub trait Space: Distribution<<Self as Space>::Element> {
    type Element;

    /// Check if the space contains a particular value
    fn contains(&self, value: &Self::Element) -> bool;

    /// Convert an element into a loggable object
    fn as_loggable(&self, element: &Self::Element) -> Loggable;
}

/// A space whose elements can be converted to feature vectors.
pub trait FeatureSpace<T, T2 = T>: Space {
    /// Length of the feature vectors in which which elements are encoded.
    fn num_features(&self) -> usize;

    /// Convert an element of the space into a feature vector.
    ///
    /// The output vector has length equal to `num_features()`.
    fn features(&self, element: &Self::Element) -> T;

    /// Construct a matrix of feature vectors for an array of elements of the space.
    ///
    /// The output is a two-dimensional array where
    /// the first dimension has length equal to `elements.len()` and
    /// the second dimension has length equal to `num_features()`.
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a;
}
