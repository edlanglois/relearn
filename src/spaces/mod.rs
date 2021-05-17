//! Spaces: runtime-defined types
mod finite;
mod index;
mod indexed_type;
mod option;
mod rl;
mod singleton;
#[cfg(test)]
pub mod testing;

pub use finite::FiniteSpace;
pub use index::IndexSpace;
pub use indexed_type::{Indexed, IndexedTypeSpace};
pub use option::OptionSpace;
pub use rl::RLSpace;
pub use singleton::SingletonSpace;

use crate::utils::distributions::BatchDistribution;
use rand::distributions::Distribution as RandDistribution;

/// A space: a set of values with some added structure.
///
/// A space is effectively a runtime-defined type.
pub trait Space {
    type Element;

    /// Check whether a particular value is contained in the space.
    fn contains(&self, value: &Self::Element) -> bool;
}

/// A space from which samples can be drawn.
pub trait SampleSpace: Space + RandDistribution<<Self as Space>::Element> {}

/// A space whose elements can be represented as value of type `T`
///
/// This representation is generally minimal, in contrast to [`FeatureSpace`],
/// which produces a representation suited for use as input to a machine learning model.
pub trait ReprSpace<T, T0 = T>: Space {
    /// Represent a single element as a scalar value.
    fn repr(&self, element: &Self::Element) -> T0;

    /// Represent a batch of elements as an array.
    fn batch_repr<'a, I>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a;
}

/// A space whose elements can be converted to feature vectors.
///
/// The representation is generally suited for use as input to a machine learning model,
/// in contrast to [`ReprSpace`], which yields a compact representation.
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

/// A space whose elements parameterize a distribution
pub trait ParameterizedDistributionSpace<T>: Space {
    /// Batched distribution type.
    ///
    /// The element representation must match the format of [`ReprSpace<_, T>`].
    /// That is, `batch_repr(&[...])` must be a valid input for [`BatchDistribution::log_probs`].
    type Distribution: BatchDistribution<T, T>;

    /// Size of the parameter vector for which elements are sampled.
    fn num_distribution_params(&self) -> usize;

    /// Sample a single element given a parameter vector.
    ///
    /// # Args
    /// * `params` - A one-dimensional parameter vector of length `self.num_distribution_params()`.
    ///
    /// # Panics
    /// Panics if `params` does not have the correct shape.
    fn sample_element(&self, params: &T) -> Self::Element;

    /// The distribution parameterized by the given parameter vector.
    ///
    /// # Args
    /// * `params` - Batched parameter vectors.
    ///              An array with shape `[BATCH_SIZE.., self.num_distribution_params()]`.
    ///
    /// # Returns
    /// The distribution(s) parameterized by `params`.
    fn distribution(&self, params: &T) -> Self::Distribution;
}

// This could possibly be a single trait like
//
// pub trait Convert<T, U> {
//     fn convert(&self, x: T) -> U;
// }
//
// But doing it for references requires higher order bounds:
// for<'a> Convert<&'a Self::Element, Foo>
// which I have had trouble getting to work in all cases.

/// Convert elements of the space into values of type `T`.
pub trait ElementInto<T>: Space {
    /// Convert an element into a value of type `T`.
    fn elem_into(&self, element: Self::Element) -> T;
}

/// Create values of type `T` from element references.
pub trait ElementRefInto<T>: Space {
    /// Create a value of type `T` from an element reference.
    fn elem_ref_into(&self, element: &Self::Element) -> T;
}

impl<T: ElementRefInto<U>, U> ElementInto<U> for T {
    fn elem_into(&self, element: Self::Element) -> U {
        self.elem_ref_into(&element)
    }
}

/// Construct elements of the space from values of type `T`
pub trait ElementFrom<T>: Space {
    /// Construct an element of the space from a value of type `T`
    fn elem_from(&self, value: T) -> Self::Element;
}

/// Try to construct an element from a value where the operation may fail.
pub trait ElementTryFrom<T>: Space {
    /// Try to construct an element from a value of type `T`, where conversion might not be possible.
    ///
    /// Returns Some(x) if and only if x is an element of this space.
    fn elem_try_from(&self, value: T) -> Option<Self::Element>;
}
