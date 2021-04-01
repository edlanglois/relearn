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

/// A space: a set of values with some added structure.
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

/// Sample elements from parameter vectors.
///
/// Uses an arbitrary non-seeded source of randomness.
pub trait ParameterizedSampleSpace<T, T2 = T>: Space {
    /// Size of the parameter vector for which elements are sampled.
    fn num_sample_params(&self) -> usize;

    /// Sample an element from a parameter vector.
    ///
    /// The input array must have length equal to `num_sample_params()`.
    fn sample(&self, parameters: &T) -> Self::Element;

    /// Log probabilities of elements under corresponding parameterized distributions.
    ///
    /// # Args
    /// * `parameters` - A two-dimensional array of shape `[N, num_sample_params()]`
    /// * `elements` - A list of N elements.
    ///
    /// # Returns
    /// A one-dimensional array of length N containing the log probability of each element
    /// under the distribution defined by the corresponding parameter vector.
    fn batch_log_probs<'a, I>(&self, parameters: &T2, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a;
}

///// Enables sampling of elements from parameter vectors.
//pub trait ParameterizedSamplingSpace<T, T2 = T>: Space {
//    /// Length of the parameter vectors from which elements are sampled.
//    fn num_sampling_parameters() -> usize;

//    /// Sample an element from a paramter vector.
//    ///
//    /// The input array must have length equal to `num_sampling_parameters()`.

//}
