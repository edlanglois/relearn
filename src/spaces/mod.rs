//! Space trait definitions
mod finite;
mod index;
mod indexed_type;
mod rl;
mod singleton;
#[cfg(test)]
pub mod testing;

pub use finite::FiniteSpace;
pub use index::IndexSpace;
pub use indexed_type::{Indexed, IndexedTypeSpace};
pub use rl::RLSpace;
pub use singleton::SingletonSpace;

use rand::distributions::Distribution;

/// A space: a set of values with some added structure.
///
/// A space is effectively a runtime-defined type.
pub trait Space {
    type Element;

    /// Check whether a particular value is contained in the space.
    fn contains(&self, value: &Self::Element) -> bool;
}

/// A space from which samples can be drawn.
pub trait SampleSpace: Space + Distribution<<Self as Space>::Element> {}

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

/// Convert elements of the space into values of type T
pub trait ElementInto<T>: Space {
    /// Convert an element into a value of type T
    fn elem_into(&self, element: Self::Element) -> T;
}

/// Create values of type T from element references.
pub trait ElementRefInto<T>: Space {
    /// Create a value of type T from an element reference.
    fn elem_ref_into(&self, element: &Self::Element) -> T;
}

impl<T: ElementRefInto<U>, U> ElementInto<U> for T {
    fn elem_into(&self, element: Self::Element) -> U {
        self.elem_ref_into(&element)
    }
}

/// Construct elements of the space from values of type T
pub trait ElementFrom<T>: Space {
    /// Construct an element of the space from a value of type T
    fn elem_from(&self, value: T) -> Self::Element;
}

/// Try to construct an element from a value where the operation may fail.
pub trait ElementTryFrom<T>: Space {
    /// Try to construct an element from a value of type T, where conversion might not be possible.
    ///
    /// Returns Some(x) if and only if x is an element of this space.
    fn elem_try_from(&self, value: T) -> Option<Self::Element>;
}
