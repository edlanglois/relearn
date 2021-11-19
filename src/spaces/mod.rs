//! Spaces: runtime-defined types
//!
//! # `PartialOrd` for Spaces
//! Spaces have a partial order such that for two spaces of the same type,
//! `a < b` means that `a` is a strict subset of `b`.
#[cfg(test)]
#[macro_use]
pub mod testing;

mod array;
mod boolean;
mod categorical;
mod index;
mod indexed_type;
mod interval;
mod nonempty_features;
mod option;
mod power;
mod product;
mod singleton;
mod wrapper;

pub use array::ArraySpace;
pub use boolean::BooleanSpace;
pub use categorical::CategoricalSpace;
pub use index::IndexSpace;
pub use indexed_type::{Indexed, IndexedTypeSpace};
pub use interval::IntervalSpace;
pub use nonempty_features::NonEmptyFeatures;
pub use option::OptionSpace;
pub use power::PowerSpace;
pub use product::ProductSpace;
pub use singleton::SingletonSpace;
pub use wrapper::BoxSpace;

// Re-export Indexed macro from relearn_derive
pub use relearn_derive::Indexed;

use crate::utils::distributions::ArrayDistribution;
use crate::utils::num_array::{BuildFromArray1D, BuildFromArray2D, NumArray1D, NumArray2D};
use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Float;
use rand::distributions::Distribution as RandDistribution;
use std::cmp::Ordering;
use std::iter::ExactSizeIterator;

/// A space: a set of values with some added structure.
///
/// A space is effectively a runtime-defined type.
pub trait Space {
    type Element;

    /// Check whether a particular value is contained in the space.
    fn contains(&self, value: &Self::Element) -> bool;
}

/// Compare this space to another in terms of the subset relation.
///
/// This is a partial order and the rules for implementing this are the same as for
/// [`PartialOrd`](std::cmp::PartialOrd). In particular,
/// the comparision must return `Some(Ordering::Equal)` if and only if `self == other`.
///
/// This is distinct from [`PartialOrd`](std::cmp::PartialOrd) so that `SubsetOrd` can be defined
/// on types that already implement `PartialOrd` in a different way (e.g. lexicographically).
/// It also avoids the confusion that might arise from using comparison operators (`<`, `>`, etc.)
/// since it is not obvious that "subset" is the relationship being used.
pub trait SubsetOrd: PartialEq<Self> {
    /// Compare using the subset relationship. This is a partial order.
    fn subset_cmp(&self, other: &Self) -> Option<Ordering>;

    /// Check if this is a strict subset of `other`.
    fn strict_subset_of(&self, other: &Self) -> bool {
        matches!(self.subset_cmp(other), Some(Ordering::Less))
    }

    /// Check if this is a subset (strict or equal) of `other`.
    fn subset_of(&self, other: &Self) -> bool {
        matches!(
            self.subset_cmp(other),
            Some(Ordering::Less | Ordering::Equal)
        )
    }

    /// Check if this is a strict superset of `other`.
    fn strict_superset_of(&self, other: &Self) -> bool {
        matches!(self.subset_cmp(other), Some(Ordering::Greater))
    }

    /// Check if this is a superset (strict or equal) of `other`.
    fn superset_of(&self, other: &Self) -> bool {
        matches!(
            self.subset_cmp(other),
            Some(Ordering::Greater | Ordering::Equal)
        )
    }
}

/// Helper function to determine the subset ordering of a product of two spaces.
///
/// Given the orderings for each of the factors, the ordering is:
/// * `Equal` if both factors are `Equal`,
/// * `Less` if both factors are `Equal` or `Less` and at least one is `Less`,
/// * `Greater` if both factors are `Equal` or `Greater` and at least one is `Greater`,
/// * `None` otherwise.
#[inline]
pub const fn product_subset_ord(a: Option<Ordering>, b: Option<Ordering>) -> Option<Ordering> {
    use Ordering::*;
    match (a, b) {
        (Some(Equal), x) => x,
        (x, Some(Equal)) => x,
        (Some(Less), Some(Less)) => Some(Less),
        (Some(Greater), Some(Greater)) => Some(Greater),
        _ => None,
    }
}

/// Helper function to determine the subset ordering of a product space with any number of factors.
///
/// Given the orderings for each of the factors, the ordering is:
/// * `Equal` if all factors are `Equal`,
/// * `Less` if all factors are `Equal` or `Less` and at least one is `Less`,
/// * `Greater` if all factors are `Equal` or `Greater` and at least one is `Greater`,
/// * `None` otherwise.
#[inline]
pub fn iter_product_subset_ord<I: IntoIterator<Item = Option<Ordering>>>(
    ord_factors: I,
) -> Option<Ordering> {
    ord_factors
        .into_iter()
        .try_fold(Ordering::Equal, |prev, cmp| {
            product_subset_ord(Some(prev), cmp)
        })
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

/// Number of features for [`EncoderFeatureSpace`] and [`FeatureSpace`].
pub trait NumFeatures {
    /// Length of the feature vectors in which elements are encoded.
    fn num_features(&self) -> usize;
}

/// Encode elements as feature vectors with the help of an encoder object.
///
/// This is intended as a helper to implement faster encoding.
/// See [`FeatureSpace`] for a similar interface without an encoder.
pub trait EncoderFeatureSpace: Space + NumFeatures {
    /// Object to help with encoding.
    type Encoder;

    /// Construct an encoder to help with encoding.
    ///
    /// The encoder is only valid for while the space is not mutated.
    /// Using the encoder with a modified space may produce incorrect results or panics,
    /// but not memory safety issues.
    fn encoder(&self) -> Self::Encoder;

    /// Encode the feature vector of an element into a mutable slice.
    ///
    /// # Args
    /// * `element` - The element to encode.
    /// * `out` - A slice of length at least `num_features()` in which the features are written.
    ///           The entire slice may be written to, even if it is longer than `num_features()`.
    /// * `zeroed` - Whether `out` is zero-initialized.
    ///              Helps avoid redundant writes for sparse feature vectors.
    /// * `encoder`- Encoder helper object.
    ///
    /// # Panics
    /// * If the slice is not large enough to fit the feature vector.
    /// * May panic if the encoder is out of date for this space object.
    fn encoder_features_out<F: Float>(
        &self,
        element: &Self::Element,
        out: &mut [F],
        zeroed: bool,
        encoder: &Self::Encoder,
    );

    /// Encode the feature vector of an element into an array.
    fn encoder_features<T>(&self, element: &Self::Element, encoder: &Self::Encoder) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        let mut array = T::Array::zeros(self.num_features());
        self.encoder_features_out(element, array.as_slice_mut(), true, encoder);
        array.into()
    }

    /// Encode the feature vectors of multiple elements into rows of a two-dimensional array.
    ///
    /// # Args
    /// * `elements` - Elements to encode.
    /// * `out` - A two-dimensional array of shape at least `[elements.len(), num_features()]`.
    ///           The entire array may be written to, even if it is larger than required.
    /// * `zeroed` - Whether `out` is zero-initialized.
    ///              Helps avoid redundant writes for sparse feature vectors.
    /// * `encoder`- Encoder helper object.
    ///
    /// # Panics
    /// * If the slice is not large enough to fit the feature vector.
    /// * May panic if the encoder is out of date for this space object.
    fn encoder_batch_features_out<'a, I, A>(
        &self,
        elements: I,
        out: &mut ArrayBase<A, Ix2>,
        zeroed: bool,
        encoder: &Self::Encoder,
    ) where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
        // Don't zip rows so that we can check whether there are too few rows.
        let mut rows = out.rows_mut().into_iter();
        for element in elements {
            let mut row = rows.next().expect("fewer rows than elements");
            self.encoder_features_out(
                element,
                row.as_slice_mut().expect("could not view row as slice"),
                zeroed,
                encoder,
            );
        }
    }

    /// Encode the feature vectors of multiple elements as rows of a two-dimensional array.
    fn encoder_batch_features<'a, I, T>(&self, elements: I, encoder: &Self::Encoder) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
        T: BuildFromArray2D,
        <T::Array as NumArray2D>::Elem: Float,
    {
        let elements = elements.into_iter();
        let mut array = T::Array::zeros((elements.len(), self.num_features()));
        self.encoder_batch_features_out(elements, &mut array.view_mut(), true, encoder);
        array.into()
    }
}

/// A space whose elements can be encoded as floating-point feature vectors.
///
/// This presents a simpler interface than `EncoderFeatureSpace` and avoids the associated trait.
pub trait FeatureSpace: Space + NumFeatures {
    /// Encode the feature vector of an element into a mutable slice.
    ///
    /// # Args
    /// * `element` - The element to encode.
    /// * `out` - A slice of length at least `num_features()` in which the features are written.
    ///           The entire slice may be written to, even if it is longer than `num_features()`.
    /// * `zeroed` - Whether `out` is zero-initialized.
    ///              Helps avoid redundant writes for sparse feature vectors.
    ///
    /// # Panics
    /// If the slice is not large enough to fit the feature vector.
    fn features_out<F: Float>(&self, element: &Self::Element, out: &mut [F], zeroed: bool);

    /// Encode the featurE Vector of an element into an array.
    fn features<T>(&self, element: &Self::Element) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        let mut array = T::Array::zeros(self.num_features());
        self.features_out(element, array.as_slice_mut(), true);
        array.into()
    }

    /// Encode the feature vectors of multiple elements into rows of a two-dimensional array.
    ///
    /// # Args
    /// * `elements` - Elements to encode.
    /// * `out` - A two-dimensional array of shape at least `[elements.len(), num_features()]`.
    ///           The entire array may be written to, even if it is larger than required.
    /// * `zeroed` - Whether `out` is zero-initialized.
    ///              Helps avoid redundant writes for sparse feature vectors.
    ///
    /// # Panics
    /// If the array is not large enough to fit the feature vectors.
    fn batch_features_out<'a, I, A>(&self, elements: I, out: &mut ArrayBase<A, Ix2>, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
        // Don't zip rows so that we can check whether there are too few rows.
        let mut rows = out.rows_mut().into_iter();
        for element in elements {
            let mut row = rows.next().expect("fewer rows than elements");
            self.features_out(
                element,
                row.as_slice_mut().expect("could not view row as slice"),
                zeroed,
            );
        }
    }

    /// Encode the feature vectors of multiple elements as rows of a two-dimensional array.
    fn batch_features<'a, I, T>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
        T: BuildFromArray2D,
        <T::Array as NumArray2D>::Elem: Float,
    {
        let elements = elements.into_iter();
        let mut array = T::Array::zeros((elements.len(), self.num_features()));
        self.batch_features_out(elements, &mut array.view_mut(), true);
        array.into()
    }
}

impl<S: EncoderFeatureSpace> FeatureSpace for S {
    fn features_out<F: Float>(&self, element: &Self::Element, out: &mut [F], zeroed: bool) {
        self.encoder_features_out(element, out, zeroed, &self.encoder())
    }
    fn features<T>(&self, element: &Self::Element) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        self.encoder_features(element, &self.encoder())
    }
    fn batch_features_out<'a, I, A>(&self, elements: I, out: &mut ArrayBase<A, Ix2>, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
        self.encoder_batch_features_out(elements, out, zeroed, &self.encoder())
    }
    fn batch_features<'a, I, T>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
        T: BuildFromArray2D,
        <T::Array as NumArray2D>::Elem: Float,
    {
        self.encoder_batch_features(elements, &self.encoder())
    }
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
