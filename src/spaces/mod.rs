//! Spaces: runtime-defined types
//!
//! # `PartialOrd` for Spaces
//! Spaces have a partial order such that for two spaces of the same type,
//! `a < b` means that `a` is a strict subset of `b`.
mod array;
mod boolean;
mod categorical;
mod index;
mod indexed_type;
mod interval;
mod nonempty_features;
mod option;
mod product;
mod singleton;
#[cfg(test)]
pub mod testing;
mod wrapper;

pub use array::ArraySpace;
pub use boolean::BooleanSpace;
pub use categorical::CategoricalSpace;
pub use index::IndexSpace;
pub use indexed_type::{Indexed, IndexedTypeSpace};
pub use interval::IntervalSpace;
pub use nonempty_features::NonEmptyFeatures;
pub use option::OptionSpace;
pub use product::ProductSpace;
pub use singleton::SingletonSpace;
pub use wrapper::BoxSpace;

// Re-export Indexed macro from relearn_derive
pub use relearn_derive::Indexed;

use crate::utils::distributions::ArrayDistribution;
use rand::distributions::Distribution as RandDistribution;
use std::iter::ExactSizeIterator;

/// A space: a set of values with some added structure.
///
/// A space is effectively a runtime-defined type.
pub trait Space {
    type Element;

    /// Check whether a particular value is contained in the space.
    fn contains(&self, value: &Self::Element) -> bool;
}

/// A space containing finitely many elements.
pub trait FiniteSpace: Space {
    /// The number of elements in the space.
    fn size(&self) -> usize;

    /// Get the index of an element.
    fn to_index(&self, element: &Self::Element) -> usize;

    /// Try to convert an index to an element.
    ///
    /// The return value is `Some(elem)` if and only if
    /// `elem` is the unique element in the space with `to_index(elem) == index`.
    fn from_index(&self, index: usize) -> Option<Self::Element>;

    /// Try to convert an index to an element.
    ///
    /// If None is returned then the index was invalid.
    /// It is allowed that Some value may be returned even if the index is invalid.
    /// If you need to validate the returned value, use [`FiniteSpace::from_index`].
    fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
        self.from_index(index)
    }
}

/// A space from which samples can be drawn.
///
/// No particular distribution is specified but the distribution:
/// * must have support equal to the entire space, and
/// * should be some form of reasonable "standard" distribution for the space.
pub trait SampleSpace: Space + RandDistribution<<Self as Space>::Element> {}

impl<S> SampleSpace for S where S: Space + RandDistribution<<Self as Space>::Element> {}

/// A space whose elements can be represented as value of type `T`
///
/// This representation is generally minimal, in contrast to [`FeatureSpace`],
/// which produces a representation suited for use as input to a machine learning model.
pub trait ReprSpace<T, T0 = T>: Space {
    /// Representation of a single element.
    fn repr(&self, element: &Self::Element) -> T0;

    /// Represent a batch of elements as an array.
    fn batch_repr<'a, I>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a;
}

/// A space whose elements can be encoded as features.
///
/// This is the base, output-type-independent feature space trait.
pub trait BaseFeatureSpace {
    /// Length of the feature vectors in which which elements are encoded.
    fn num_features(&self) -> usize;
}

/// A space whose elements can be converted to feature vectors.
///
/// The representation is generally suited for use as input to a machine learning model,
/// in contrast to [`ReprSpace`], which yields a compact representation.
pub trait FeatureSpace<T>: Space + BaseFeatureSpace {
    /// Convert an element of the space into a feature vector.
    ///
    /// # Args
    /// * `element` - An element of the space.
    ///
    /// # Returns
    /// A feature vector with length `NUM_FEATURES`.
    fn features(&self, element: &Self::Element) -> T;
}

/// A space whose elements can written into an array as feature vectors.
///
/// The representation is generally suited for use as input to a machine learning model,
/// in contrast to [`ReprSpace`], which yields a compact representation.
pub trait FeatureSpaceOut<T>: Space + BaseFeatureSpace {
    /// Convert an element of the space into a feature vector.
    ///
    /// # Args
    /// * `element` - An element of the space.
    /// * `out`     - A vector with length `NUM_FEATURES` into which the features are written.
    /// * `zeroed`  - Whether `out` contains nothing but zeros. For spaces with sparse features,
    ///             knowing this avoids having to write most of the array.
    fn features_out(&self, element: &Self::Element, out: &mut T, zeroed: bool);
}

/// A space whose elements can be converted to feature vectors in a batch.
pub trait BatchFeatureSpace<T2>: Space + BaseFeatureSpace {
    /// Construct a matrix of feature vectors for a set of elements.
    ///
    /// # Args
    /// * `elements` - A set of elements of the space.
    ///
    /// # Returns
    /// A two-dimensional array of shape `[NUM_ELEMENTS, NUM_FEATURES]`.
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a;
}

/// A space whose elements can written into an array as feature vectors in a batch.
///
/// The representation is generally suited for use as input to a machine learning model,
/// in contrast to [`ReprSpace`], which yields a compact representation.
pub trait BatchFeatureSpaceOut<T2>: Space + BaseFeatureSpace {
    /// Construct a matrix of feature vectors for a set of elements.
    ///
    /// # Args
    /// * `elements` - A set of elements of the space.
    ///
    /// * `out`      - A two-dimensional array of shape `[NUM_ELEMENTS, NUM_FEATURES]`
    ///              into which the features are written.
    ///
    /// * `zeroed`   - Whether `out` contains nothing but zeros. For spaces with sparse features,
    ///              knowing this avoids having to write most of the array.
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut T2, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: Clone,
        Self::Element: 'a;
}

/// A space whose elements parameterize a distribution
pub trait ParameterizedDistributionSpace<T, T2 = T>: ReprSpace<T, T2> {
    /// Batched distribution type.
    ///
    /// The element representation must match the format of [`ReprSpace`].
    /// That is, `batch_repr(&[...])` must be a valid input for [`ArrayDistribution::log_probs`].
    type Distribution: ArrayDistribution<T, T>;

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
    fn distribution(&self, params: &T2) -> Self::Distribution;
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

/// Space that enforces `Space::Element: Send`
///
/// This trait uses an associated type because directly bounding `where Space::Element: Send`
/// does not stick to the trait. It must instead be re-asserted on each use.
/// See <https://stackoverflow.com/questions/37600687>.
///
pub trait SendElementSpace: Space<Element = Self::SendElement> {
    type SendElement: Send;
}
impl<S> SendElementSpace for S
where
    S: Space,
    Self::Element: Send,
{
    type SendElement = Self::Element;
}
