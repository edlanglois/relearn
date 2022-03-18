//! `IndexedTypeSpace` and `Indexed` trait
use super::{
    ElementRefInto, FeatureSpace, FiniteSpace, NonEmptySpace, ParameterizedDistributionSpace,
    ReprSpace, Space, SubsetOrd,
};
use crate::logging::Loggable;
use crate::torch::distributions::Categorical;
use crate::utils::distributions::ArrayDistribution;
use ndarray::{s, ArrayBase, DataMut, Ix2};
use num_traits::{Float, One, Zero};
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::{any, fmt};
use tch::{Device, Kind, Tensor};

/// An indexed set of finitely many possibilities.
///
/// Can be implemented automatically for enum types with no internal data
/// using `#[derive(Indexed)]`.
///
/// ```
/// use relearn::spaces::Indexed;
///
/// #[derive(Indexed)]
/// enum Foo {
///     A,
///     B,
/// }
///
/// assert_eq!(Foo::SIZE, 2);
/// assert_eq!(Foo::B.index(), 1);
/// ```
pub trait Indexed {
    /// The number of possible values this type can represent.
    const SIZE: usize;

    /// The index of this element.
    fn index(&self) -> usize;

    /// Construct from an index.
    fn from_index(index: usize) -> Option<Self>
    where
        Self: Sized;
}

/// A space defined over an indexed type.
///
/// The wrapped type must implement [`Indexed`].
/// Use `#[derive(Indexed)]` to implement `Indexed` automatically for enum types that have no
/// internal data.
// Can only use derives for serde traits
// because they allow modifying the bounds to remove `T: <Trait>`
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IndexedTypeSpace<T> {
    // <fn() -> T> allows Sync and Send without adding a drop check
    // https://stackoverflow.com/a/50201389/1267562
    #[serde(skip)]
    element_type: PhantomData<fn() -> T>,
}

impl<T> IndexedTypeSpace<T> {
    // Cannot be const because
    // E0658: function pointers cannot appear in constant functions
    #[allow(clippy::missing_const_for_fn)]
    pub fn new() -> Self {
        Self {
            element_type: PhantomData,
        }
    }
}

impl<T> Default for IndexedTypeSpace<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Debug for IndexedTypeSpace<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IndexedTypeSpace<{}>", any::type_name::<T>())
    }
}

impl<T> fmt::Display for IndexedTypeSpace<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IndexedTypeSpace<{}>", any::type_name::<T>())
    }
}

impl<T> Clone for IndexedTypeSpace<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T> Copy for IndexedTypeSpace<T> {}

impl<T: Clone + Send> Space for IndexedTypeSpace<T> {
    type Element = T;

    fn contains(&self, _element: &Self::Element) -> bool {
        true
    }
}

impl<T> PartialEq for IndexedTypeSpace<T> {
    fn eq(&self, _other: &Self) -> bool {
        true // There is only one kind of IndexedTypeSpace<T>
    }
}

impl<T> Eq for IndexedTypeSpace<T> {}

impl<T> Hash for IndexedTypeSpace<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<T> SubsetOrd for IndexedTypeSpace<T> {
    fn subset_cmp(&self, _other: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl<T: Indexed + Clone + Send> NonEmptySpace for IndexedTypeSpace<T> {
    fn some_element(&self) -> Self::Element {
        T::from_index(0).expect("space is empty")
    }
}

impl<T: Indexed> Distribution<T> for IndexedTypeSpace<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        T::from_index(rng.gen_range(0..T::SIZE)).unwrap()
    }
}

impl<T: Indexed + Clone + Send> FiniteSpace for IndexedTypeSpace<T> {
    fn size(&self) -> usize {
        T::SIZE
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        T::index(element)
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        T::from_index(index)
    }
}

/// Features are one-hot vectors
impl<T: Indexed + Clone + Send> FeatureSpace for IndexedTypeSpace<T> {
    #[inline]
    fn num_features(&self) -> usize {
        T::SIZE
    }

    #[inline]
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        zeroed: bool,
    ) -> &'a mut [F] {
        let (out, rest) = out.split_at_mut(T::SIZE);
        if !zeroed {
            out.fill(F::zero());
        }
        out[self.to_index(element)] = F::one();
        rest
    }

    #[inline]
    fn batch_features_out<'a, I, A>(&self, elements: I, out: &mut ArrayBase<A, Ix2>, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
        if !zeroed {
            out.slice_mut(s![.., 0..self.num_features()])
                .fill(Zero::zero());
        }

        // Don't zip rows so that we can check whether there are too few rows.
        let mut rows = out.rows_mut().into_iter();
        for element in elements {
            let mut row = rows.next().expect("fewer rows than elements");
            row[self.to_index(element)] = One::one();
        }
    }
}

/// Represents elements as integer tensors.
impl<T: Indexed + Clone + Send> ReprSpace<Tensor> for IndexedTypeSpace<T> {
    fn repr(&self, element: &Self::Element) -> Tensor {
        Tensor::scalar_tensor(self.to_index(element) as i64, (Kind::Int64, Device::Cpu))
    }

    fn batch_repr<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let indices: Vec<_> = elements
            .into_iter()
            .map(|elem| self.to_index(elem) as i64)
            .collect();
        Tensor::of_slice(&indices)
    }
}

impl<T: Indexed + Clone + Send> ParameterizedDistributionSpace<Tensor> for IndexedTypeSpace<T> {
    type Distribution = Categorical;

    fn num_distribution_params(&self) -> usize {
        T::SIZE
    }

    fn sample_element(&self, params: &Tensor) -> Self::Element {
        self.from_index(
            self.distribution(params)
                .sample()
                .int64_value(&[])
                .try_into()
                .unwrap(),
        )
        .unwrap()
    }

    fn distribution(&self, params: &Tensor) -> Self::Distribution {
        Self::Distribution::new(params)
    }
}

/// Log the index as a sample from `0..N`
impl<T: Indexed + Clone + Send> ElementRefInto<Loggable> for IndexedTypeSpace<T> {
    fn elem_ref_into(&self, element: &Self::Element) -> Loggable {
        Loggable::Index {
            value: self.to_index(element),
            size: T::SIZE,
        }
    }
}

impl Indexed for bool {
    const SIZE: usize = 2;

    fn index(&self) -> usize {
        (*self).into()
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

#[cfg(test)]
mod trit {
    use relearn_derive::Indexed;

    #[derive(Debug, Copy, Clone, Indexed, PartialEq, Eq)]
    pub enum Trit {
        Zero,
        One,
        Two,
    }
}

#[cfg(test)]
mod space {
    use super::super::testing;
    use super::trit::Trit;
    use super::*;

    fn check_contains_samples<T: Indexed + Clone + Send>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_contains_samples(&space, 100);
    }

    #[test]
    fn contains_samples_bool() {
        check_contains_samples::<bool>();
    }

    #[test]
    fn contains_samples_enum() {
        check_contains_samples::<Trit>();
    }

    fn check_from_to_index_iter_size<T: Indexed + Clone + Send>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_iter_size_bool() {
        check_from_to_index_iter_size::<bool>();
    }

    #[test]
    fn from_to_index_iter_size_enum() {
        check_from_to_index_iter_size::<Trit>();
    }

    fn check_from_index_sampled<T: Indexed + Clone + Send>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_from_index_sampled(&space, 20);
    }

    #[test]
    fn from_index_sampled_bool() {
        check_from_index_sampled::<bool>();
    }

    #[test]
    fn from_index_sampled_enum() {
        check_from_index_sampled::<Trit>();
    }

    fn check_from_index_invalid<T: Indexed + Clone + Send>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn from_index_invalid_bool() {
        check_from_index_invalid::<bool>();
    }

    #[test]
    fn from_index_invalid_enum() {
        check_from_index_invalid::<Trit>();
    }
}

#[cfg(test)]
mod subset_ord {
    use super::super::SubsetOrd;
    use super::trit::Trit;
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn eq() {
        assert_eq!(
            IndexedTypeSpace::<Trit>::new(),
            IndexedTypeSpace::<Trit>::new()
        );
    }

    #[test]
    fn cmp_equal() {
        assert_eq!(
            IndexedTypeSpace::<Trit>::new().subset_cmp(&IndexedTypeSpace::<Trit>::new()),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn not_strict_subset() {
        assert!(
            !(IndexedTypeSpace::<Trit>::new().strict_subset_of(&IndexedTypeSpace::<Trit>::new()))
        );
    }
}

#[cfg(test)]
mod serialize {
    use super::trit::Trit;
    use super::*;
    use serde_test::{assert_tokens, Token};

    #[test]
    fn ser_de_tokens() {
        let space = IndexedTypeSpace::<Trit>::new();
        assert_tokens(
            &space,
            &[
                Token::Struct {
                    name: "IndexedTypeSpace",
                    len: 0,
                },
                Token::StructEnd,
            ],
        );
    }
}

#[cfg(test)]
/// Test the `#[derive(Indexed)]` macro
mod derive_indexed_macro {
    use super::*;
    use relearn_derive::Indexed;

    #[derive(Debug, Indexed)]
    enum EmptyEnum {}

    #[derive(Debug, Indexed)]
    enum NonEmptyEnum {
        A,
        B,
    }

    #[test]
    fn empty_enum_len() {
        assert_eq!(EmptyEnum::SIZE, 0);
    }

    #[test]
    fn empty_enum_from_index_invalid_0() {
        let result = EmptyEnum::from_index(0);
        assert!(result.is_none(), "Expected `None`, got {:?}", result);
    }

    #[test]
    fn empty_enum_from_index_invalid_1() {
        let result = EmptyEnum::from_index(1);
        assert!(result.is_none(), "Expected `None`, got {:?}", result);
    }

    #[test]
    fn non_empty_enum_len() {
        assert_eq!(NonEmptyEnum::SIZE, 2);
    }

    #[test]
    fn non_empty_enum_to_index() {
        assert_eq!(NonEmptyEnum::A.index(), 0);
        assert_eq!(NonEmptyEnum::B.index(), 1);
    }

    #[test]
    fn non_empty_enum_from_index_valid_0() {
        let result = NonEmptyEnum::from_index(0);
        if let Some(NonEmptyEnum::A) = result {
        } else {
            panic!("Expected `Some(NonEmptyEnum::A)`, got {:?}", result);
        }
    }

    #[test]
    fn non_empty_enum_from_index_valid_1() {
        let result = NonEmptyEnum::from_index(1);
        if let Some(NonEmptyEnum::B) = result {
        } else {
            panic!("Expected `Some(NonEmptyEnum::B)`, got {:?}", result);
        }
    }

    #[test]
    fn non_empty_enum_from_index_invalid_2() {
        let result = NonEmptyEnum::from_index(2);
        assert!(result.is_none(), "Expected `None`, got {:?}", result);
    }
}

#[cfg(test)]
mod feature_space {
    use super::trit::Trit;
    use super::*;

    fn space() -> IndexedTypeSpace<Trit> {
        IndexedTypeSpace::new()
    }

    #[test]
    fn num_features() {
        let space = space();
        assert_eq!(3, space.num_features());
    }

    features_tests!(trit_zero, space(), Trit::Zero, [1.0, 0.0, 0.0]);
    features_tests!(trit_one, space(), Trit::One, [0.0, 1.0, 0.0]);
    features_tests!(trit_two, space(), Trit::Two, [0.0, 0.0, 1.0]);
    batch_features_tests!(
        trit_batch,
        space(),
        [Trit::Two, Trit::Zero, Trit::One, Trit::Zero],
        [
            [0.0, 0.0, 1.0], // Two
            [1.0, 0.0, 0.0], // Zero
            [0.0, 1.0, 0.0], // One
            [1.0, 0.0, 0.0]  // Zero
        ]
    );
}

#[cfg(test)]
mod repr_space_tensor {
    use super::trit::Trit;
    use super::*;

    #[test]
    fn repr() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        assert_eq!(
            space.repr(&Trit::Zero),
            Tensor::scalar_tensor(0, (Kind::Int64, Device::Cpu))
        );
        assert_eq!(
            space.repr(&Trit::One),
            Tensor::scalar_tensor(1, (Kind::Int64, Device::Cpu))
        );
        assert_eq!(
            space.repr(&Trit::Two),
            Tensor::scalar_tensor(2, (Kind::Int64, Device::Cpu))
        );
    }

    #[test]
    fn batch_repr() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let elements = [Trit::Zero, Trit::One, Trit::Two, Trit::One];
        let actual = space.batch_repr(&elements);
        let expected = Tensor::of_slice(&[0_i64, 1, 2, 1]);
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod parameterized_sample_space_tensor {
    use super::super::IndexedTypeSpace;
    use super::trit::Trit;
    use super::*;

    #[test]
    fn num_sample_params() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        assert_eq!(3, space.num_distribution_params());
    }

    #[test]
    fn sample_element_deterministic() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let params = Tensor::of_slice(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]);
        for _ in 0..10 {
            assert_eq!(Trit::One, space.sample_element(&params));
        }
    }

    #[test]
    fn sample_element_two_of_three() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        let params = Tensor::of_slice(&[f32::NEG_INFINITY, 0.0, 0.0]);
        for _ in 0..10 {
            assert!(Trit::Zero != space.sample_element(&params));
        }
    }

    #[test]
    fn sample_element_check_distribution() {
        let space: IndexedTypeSpace<Trit> = IndexedTypeSpace::new();
        // Probabilities: [0.09, 0.24, 0.67]
        let params = Tensor::of_slice(&[-1.0, 0.0, 1.0]);
        let mut one_count = 0;
        let mut two_count = 0;
        let mut three_count = 0;
        for _ in 0..1000 {
            match space.sample_element(&params) {
                Trit::Zero => one_count += 1,
                Trit::One => two_count += 1,
                Trit::Two => three_count += 1,
            }
        }
        // Check that the counts are within 3.5 standard deviations of the mean
        assert!((58..=121).contains(&one_count));
        assert!((197..=292).contains(&two_count));
        assert!((613..=717).contains(&three_count));
    }
}
