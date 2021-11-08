//! `IndexedTypeSpace` and `Indexed` trait
use super::{CategoricalSpace, FiniteSpace, Space};
use rand::distributions::Distribution;
use rand::Rng;
use std::any;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

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
/// assert_eq!(Foo::B.as_index(), 1);
/// ```
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
///
/// The wrapped type must implement [`Indexed`].
/// Use `#[derive(Indexed)]` to implement `Indexed` automatically for enum types that have no
/// internal data.
pub struct IndexedTypeSpace<T> {
    // <fn(T) -> T> allows Sync and Send without adding a drop check
    // https://stackoverflow.com/a/50201389/1267562
    element_type: PhantomData<fn(T) -> T>,
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

impl<T> Space for IndexedTypeSpace<T> {
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

impl<T> PartialOrd for IndexedTypeSpace<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for IndexedTypeSpace<T> {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl<T> Hash for IndexedTypeSpace<T> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

// Subspaces
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

impl<T: Indexed> CategoricalSpace for IndexedTypeSpace<T> {}

impl Indexed for bool {
    const SIZE: usize = 2;

    fn as_index(&self) -> usize {
        *self as usize
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
mod space {
    use super::super::testing;
    use super::*;

    enum TestEnum {
        A,
        B,
        C,
    }

    impl Indexed for TestEnum {
        const SIZE: usize = 3;
        fn as_index(&self) -> usize {
            match self {
                TestEnum::A => 0,
                TestEnum::B => 1,
                TestEnum::C => 2,
            }
        }

        fn from_index(index: usize) -> Option<Self> {
            match index {
                0 => Some(Self::A),
                1 => Some(Self::B),
                2 => Some(Self::C),
                _ => None,
            }
        }
    }

    fn check_contains_samples<T: Indexed>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_contains_samples(&space, 100);
    }

    #[test]
    fn contains_samples_bool() {
        check_contains_samples::<bool>();
    }

    #[test]
    fn contains_samples_enum() {
        check_contains_samples::<TestEnum>();
    }

    fn check_from_to_index_iter_size<T: Indexed>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_iter_size_bool() {
        check_from_to_index_iter_size::<bool>();
    }

    #[test]
    fn from_to_index_iter_size_enum() {
        check_from_to_index_iter_size::<TestEnum>();
    }

    fn check_from_index_sampled<T: Indexed>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_from_index_sampled(&space, 100);
    }

    #[test]
    fn from_index_sampled_bool() {
        check_from_index_sampled::<bool>();
    }

    #[test]
    fn from_index_sampled_enum() {
        check_from_index_sampled::<TestEnum>();
    }

    fn check_from_index_invalid<T: Indexed>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn from_index_invalid_bool() {
        check_from_index_invalid::<bool>();
    }

    #[test]
    fn from_index_invalid_enum() {
        check_from_index_invalid::<TestEnum>();
    }
}

#[cfg(test)]
mod partial_ord {
    use super::*;
    use relearn_derive::Indexed;

    #[derive(Debug, Indexed)]
    enum TestEnum {
        A,
        B,
        C,
    }

    #[test]
    fn eq() {
        assert_eq!(
            IndexedTypeSpace::<TestEnum>::new(),
            IndexedTypeSpace::<TestEnum>::new()
        );
    }

    #[test]
    fn cmp_equal() {
        assert_eq!(
            IndexedTypeSpace::<TestEnum>::new().cmp(&IndexedTypeSpace::<TestEnum>::new()),
            Ordering::Equal
        );
    }

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn not_less() {
        assert!(!(IndexedTypeSpace::<TestEnum>::new() < IndexedTypeSpace::<TestEnum>::new()));
    }
}

#[cfg(test)]
#[allow(clippy::missing_const_for_fn)]
/// Test the derive(Indexed) macro
mod finite_space {
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
        assert_eq!(NonEmptyEnum::A.as_index(), 0);
        assert_eq!(NonEmptyEnum::B.as_index(), 1);
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
