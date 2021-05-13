//! IndexedTypeSpace and Indexed trait
use super::{ElementRefInto, FiniteSpace, SampleSpace, Space};

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
#[derive(Clone)]
pub struct IndexedTypeSpace<T: Indexed> {
    element_type: PhantomData<T>,
}

impl<T: Indexed> IndexedTypeSpace<T> {
    pub fn new() -> Self {
        Self {
            element_type: PhantomData,
        }
    }
}

impl<T: Indexed> Default for IndexedTypeSpace<T> {
    fn default() -> Self {
        Self::new()
    }
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
}

// Subspaces
impl<T: Indexed> Distribution<T> for IndexedTypeSpace<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        T::from_index(rng.gen_range(0, T::SIZE)).unwrap()
    }
}

impl<T: Indexed> SampleSpace for IndexedTypeSpace<T> {}

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

impl<T: Indexed> ElementRefInto<Loggable> for IndexedTypeSpace<T> {
    fn elem_ref_into(&self, element: &Self::Element) -> Loggable {
        Loggable::IndexSample {
            value: T::as_index(element),
            size: T::SIZE,
        }
    }
}

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
mod indexed_type_space {
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
                0 => Some(TestEnum::A),
                1 => Some(TestEnum::B),
                2 => Some(TestEnum::C),
                _ => None,
            }
        }
    }

    fn check_contains_samples<T: Indexed>() {
        let space = IndexedTypeSpace::<T>::new();
        testing::check_contains_samples(space, 100);
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
        testing::check_from_to_index_iter_size(space);
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
        testing::check_from_index_sampled(space, 100);
    }

    #[test]
    fn from_index_sampled_bool() {
        check_from_index_sampled::<bool>();
    }

    #[test]
    fn from_index_sampled_enum() {
        check_from_index_sampled::<TestEnum>();
    }
}

#[cfg(test)]
/// Test the derive(Indexed) macro
mod derive_indexed_tests {
    use super::*;
    use rust_rl_derive::Indexed;

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
