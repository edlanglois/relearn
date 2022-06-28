//! Spaces: runtime-defined types
//!
//! In addition to the spaces defined here,
//! a product space can be derived on structures containing inner spaces with
//! [`#[derive(ProductSpace)]`](ProductSpace).
#[cfg(test)]
#[macro_use]
pub mod testing;

mod array;
mod boolean;
mod index;
mod indexed_type;
mod interval;
mod nonempty_features;
mod option;
mod power;
mod singleton;
#[cfg(test)]
mod test_derive;
mod tuple;
mod wrapper;

pub use array::ArraySpace;
pub use boolean::BooleanSpace;
pub use index::IndexSpace;
pub use indexed_type::{Indexed, IndexedTypeSpace};
pub use interval::IntervalSpace;
pub use nonempty_features::NonEmptyFeatures;
pub use option::OptionSpace;
pub use power::PowerSpace;
pub use singleton::SingletonSpace;
pub use tuple::{TupleSpace2, TupleSpace3, TupleSpace4, TupleSpace5};
pub use wrapper::BoxSpace;

// Re-export space macros from relearn_derive
pub use relearn_derive::{
    FiniteSpace, Indexed, LogElementSpace, ProductSpace, SampleSpace, Space, SubsetOrd,
};

use crate::logging::{LogError, StatsLogger};
use crate::utils::distributions::ArrayDistribution;
use crate::utils::num_array::{BuildFromArray1D, BuildFromArray2D, NumArray1D, NumArray2D};
use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Float;
use rand::distributions::Distribution;
use rand::RngCore;
use std::cmp::Ordering;
use std::iter::ExactSizeIterator;

/// A space: a set of values with some added structure.
///
/// A space is effectively a runtime-defined type.
pub trait Space {
    // It is awkward to constrain associated types in sub-traits so apply core constraints here.
    type Element: Clone + Send;

    /// Check whether a particular value is contained in the space.
    fn contains(&self, value: &Self::Element) -> bool;
}

/// Implement `Space` for a deref-able wrapper type generic over `S: Space + ?Sized`.
macro_rules! impl_wrapped_space {
    ($wrapper:ty) => {
        impl<S> Space for $wrapper
        where
            S: Space + ?Sized,
        {
            type Element = S::Element;

            #[inline]
            fn contains(&self, value: &Self::Element) -> bool {
                S::contains(self, value)
            }
        }
    };
}
impl_wrapped_space!(&'_ S);
impl_wrapped_space!(Box<S>);

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
    #[inline]
    fn strict_subset_of(&self, other: &Self) -> bool {
        matches!(self.subset_cmp(other), Some(Ordering::Less))
    }

    /// Check if this is a subset (strict or equal) of `other`.
    #[inline]
    fn subset_of(&self, other: &Self) -> bool {
        matches!(
            self.subset_cmp(other),
            Some(Ordering::Less | Ordering::Equal)
        )
    }

    /// Check if this is a strict superset of `other`.
    #[inline]
    fn strict_superset_of(&self, other: &Self) -> bool {
        matches!(self.subset_cmp(other), Some(Ordering::Greater))
    }

    /// Check if this is a superset (strict or equal) of `other`.
    #[inline]
    fn superset_of(&self, other: &Self) -> bool {
        matches!(
            self.subset_cmp(other),
            Some(Ordering::Greater | Ordering::Equal)
        )
    }
}

/// Implement `SubsetOrd` for a deref-able wrapper type generic over `T: SubsetOrd + ?Sized`.
macro_rules! impl_wrapped_subset_ord {
    ($wrapper:ty) => {
        impl<S> SubsetOrd for $wrapper
        where
            S: SubsetOrd + ?Sized,
        {
            #[inline]
            fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
                S::subset_cmp(self, other)
            }
        }
    };
}
impl_wrapped_subset_ord!(&'_ S);
impl_wrapped_subset_ord!(Box<S>);

/// Helper function to determine the subset ordering of a product of two spaces.
///
/// Given the orderings for each of the factors, the ordering is:
/// * `Equal` if both factors are `Equal`,
/// * `Less` if both factors are `Equal` or `Less` and at least one is `Less`,
/// * `Greater` if both factors are `Equal` or `Greater` and at least one is `Greater`,
/// * `None` otherwise.
#[must_use]
#[inline]
pub const fn product_subset_ord(a: Ordering, b: Option<Ordering>) -> Option<Ordering> {
    use Ordering::*;
    match (a, b) {
        (Equal, Some(x)) | (x, Some(Equal)) => Some(x),
        (Less, Some(Less)) => Some(Less),
        (Greater, Some(Greater)) => Some(Greater),
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
        .try_fold(Ordering::Equal, product_subset_ord)
}

/// A space containing finitely many elements.
pub trait FiniteSpace: Space {
    /// The number of elements in the space.
    fn size(&self) -> usize;

    /// Get the (unique) index of an element.
    fn to_index(&self, element: &Self::Element) -> usize;

    /// Try to convert an index to an element.
    ///
    /// The return value is `Some(elem)` if and only if
    /// `elem` is the unique element in the space with `to_index(elem) == index`.
    #[allow(clippy::wrong_self_convention)] // `from_` here refers to element, not space
    fn from_index(&self, index: usize) -> Option<Self::Element>;

    /// Try to convert an index to an element.
    ///
    /// If `None` is returned then the index was invalid.
    /// `Some(_)` may be returned even if the index is invalid.
    /// If the returned value must be validated then use [`FiniteSpace::from_index`].
    #[inline]
    #[allow(clippy::wrong_self_convention)] // `from_` here refers to element, not space
    fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
        self.from_index(index)
    }
}

/// Implement `FiniteSpace` for a deref-able wrapper type generic over `S: FiniteSpace + ?Sized`.
macro_rules! impl_wrapped_finite_space {
    ($wrapper:ty) => {
        impl<S> FiniteSpace for $wrapper
        where
            S: FiniteSpace + ?Sized,
        {
            #[inline]
            fn size(&self) -> usize {
                S::size(self)
            }
            #[inline]
            fn to_index(&self, element: &Self::Element) -> usize {
                S::to_index(self, element)
            }
            #[inline]
            fn from_index(&self, index: usize) -> Option<Self::Element> {
                S::from_index(self, index)
            }
            #[inline]
            fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
                S::from_index_unchecked(self, index)
            }
        }
    };
}
impl_wrapped_finite_space!(&'_ S);
impl_wrapped_finite_space!(Box<S>);

/// A space containing at least one element.
pub trait NonEmptySpace: Space {
    /// An arbitrary deterministic element from the space.
    fn some_element(&self) -> Self::Element;
}

/// Implement `NonEmptySpace` for a deref-able wrapper type generic on `S: NonEmptySpace + ?Sized`.
macro_rules! impl_wrapped_non_empty_space {
    ($wrapper:ty) => {
        impl<S> NonEmptySpace for $wrapper
        where
            S: NonEmptySpace + ?Sized,
        {
            #[inline]
            fn some_element(&self) -> Self::Element {
                S::some_element(self)
            }
        }
    };
}
impl_wrapped_non_empty_space!(&'_ S);
impl_wrapped_non_empty_space!(Box<S>);

/// A space from which samples can be drawn.
///
/// No particular distribution is specified but the distribution:
/// * must have support equal to the entire space, and
/// * should be some form of reasonable "standard" distribution for the space.
///
/// # Note
/// This re-implements sample method of [`Distribution`] rather than set
/// `Distribution<Self::Element>` as a super-trait so that `SampleSpace` is object-safe since
/// * `Distribution<T>` is not object-safe, and even if it was,
/// * generic super traits using `<Self::AssocType>` are not object safe due to a bug / issue:
///     <https://github.com/rust-lang/rust/issues/40533>.
pub trait SampleSpace: NonEmptySpace {
    /// Sample a random element.
    fn sample(&self, rng: &mut dyn RngCore) -> Self::Element;
}

impl<S> SampleSpace for S
where
    S: NonEmptySpace + Distribution<<Self as Space>::Element>,
{
    #[inline]
    fn sample(&self, rng: &mut dyn RngCore) -> Self::Element {
        Distribution::sample(&self, rng)
    }
}

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

/// Implement `ReprSpace<T, T0>` for a deref-able wrapper type generic over `S`.
macro_rules! impl_wrapped_repr_space {
    ($wrapper:ty) => {
        impl<S, T, T0> ReprSpace<T, T0> for $wrapper
        where
            S: ReprSpace<T, T0> + ?Sized,
        {
            #[inline]
            fn repr(&self, element: &Self::Element) -> T0 {
                S::repr(self, element)
            }
            #[inline]
            fn batch_repr<'a, I>(&self, elements: I) -> T
            where
                I: IntoIterator<Item = &'a Self::Element>,
                I::IntoIter: ExactSizeIterator + Clone,
                Self::Element: 'a,
            {
                S::batch_repr(self, elements)
            }
        }
    };
}
impl_wrapped_repr_space!(&'_ S);
impl_wrapped_repr_space!(Box<S>);

/// A space whose elements can be encoded as floating-point feature vectors.
pub trait FeatureSpace: Space {
    /// Length of the encoded feature vectors.
    fn num_features(&self) -> usize;

    /// Encode the feature vector of an element into a mutable slice.
    ///
    /// # Args
    /// * `element` - The element to encode.
    /// * `out` - A slice of length at least `num_features()` in which the features are written.
    ///           Only the first `num_features()` values are written to.
    /// * `zeroed` - Whether `out` is zero-initialized.
    ///              Helps avoid redundant writes for sparse feature vectors.
    ///
    /// # Returns
    /// A reference to the remainder of out: `&mut out[num_features()..]`.
    ///
    /// # Panics
    /// If the slice is not large enough to fit the feature vector.
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        zeroed: bool,
    ) -> &'a mut [F];

    /// Encode the feature vector of an element into an array.
    #[inline]
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
    ///           Only the left `[.., 0..num_features()]` subarray may be written to.
    /// * `zeroed` - Whether `out` is zero-initialized.
    ///              Helps avoid redundant writes for sparse feature vectors.
    ///
    /// # Panics
    /// If the array is not large enough to fit the feature vectors.
    #[inline]
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
    #[inline]
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

/// A space whose elements parameterize a distribution
pub trait ParameterizedDistributionSpace<T, T2 = T>: ReprSpace<T, T2> {
    /// Batched distribution type.
    ///
    /// The element representation must match the format of [`ReprSpace`].
    /// That is, `batch_repr(&[...])` must be a valid input for [`ArrayDistribution::log_probs`].
    type Distribution: ArrayDistribution<T, T>;

    /// Size of the parameter vector for which elements are sampled.
    fn num_distribution_params(&self) -> usize;

    // TODO Take Prng?
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

/// A space whose elements can be logged to a [`StatsLogger`]
pub trait LogElementSpace: Space {
    /// Log an element of the space
    fn log_element<L: StatsLogger + ?Sized>(
        &self,
        name: &'static str,
        element: &Self::Element,
        logger: &mut L,
    ) -> Result<(), LogError>;
}
