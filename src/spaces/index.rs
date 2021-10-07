//! `IndexSpace` definition
use super::{CategoricalSpace, FiniteSpace, Space};
use rand::distributions::Distribution;
use rand::Rng;
use std::cmp::Ordering;
use std::fmt;

/// An index space; consists of the integers `0` to `size - 1`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexSpace {
    pub size: usize,
}

impl IndexSpace {
    pub const fn new(size: usize) -> Self {
        Self { size }
    }
}

impl fmt::Display for IndexSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IndexSpace({})", self.size)
    }
}

impl Space for IndexSpace {
    type Element = usize;

    fn contains(&self, value: &Self::Element) -> bool {
        value < &self.size
    }
}

impl PartialOrd for IndexSpace {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexSpace {
    fn cmp(&self, other: &Self) -> Ordering {
        self.size.cmp(&other.size)
    }
}

// Subspaces
impl Distribution<<Self as Space>::Element> for IndexSpace {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        rng.gen_range(0..self.size)
    }
}

impl FiniteSpace for IndexSpace {
    fn size(&self) -> usize {
        self.size
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        *element
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        if index >= self.size {
            None
        } else {
            Some(index)
        }
    }

    fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
        Some(index)
    }
}

impl<T: FiniteSpace + ?Sized> From<&T> for IndexSpace {
    fn from(space: &T) -> Self {
        Self { size: space.size() }
    }
}

impl CategoricalSpace for IndexSpace {}

#[cfg(test)]
mod space {
    use super::super::testing;
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn contains_zero(#[values(1, 5)] size: usize) {
        let space = IndexSpace::new(size);
        assert!(space.contains(&0));
    }

    #[rstest]
    fn not_contains_too_large(#[values(1, 5)] size: usize) {
        let space = IndexSpace::new(size);
        assert!(!space.contains(&100));
    }

    #[rstest]
    fn contains_samples(#[values(1, 5)] size: usize) {
        let space = IndexSpace::new(size);
        testing::check_contains_samples(&space, 100);
    }
}

#[cfg(test)]
mod partial_ord {
    use super::*;

    #[test]
    fn same_eq() {
        assert_eq!(IndexSpace::new(2), IndexSpace::new(2));
    }

    #[test]
    fn same_cmp_equal() {
        assert_eq!(IndexSpace::new(2).cmp(&IndexSpace::new(2)), Ordering::Equal);
    }

    #[test]
    fn different_not_eq() {
        assert!(IndexSpace::new(2) != IndexSpace::new(1));
    }

    #[test]
    fn same_le() {
        assert!(IndexSpace::new(2) <= IndexSpace::new(2));
    }

    #[test]
    fn smaller_lt() {
        assert!(IndexSpace::new(1) < IndexSpace::new(2));
    }

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn larger_not_lt() {
        assert!(!(IndexSpace::new(3) < IndexSpace::new(1)));
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::testing;
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn from_to_index_iter_size(#[values(1, 5)] size: usize) {
        let space = IndexSpace::new(size);
        testing::check_from_to_index_iter_size(&space);
    }

    #[rstest]
    fn from_index_sampled(#[values(1, 5)] size: usize) {
        let space = IndexSpace::new(size);
        testing::check_from_index_sampled(&space, 100);
    }

    #[rstest]
    fn from_index_invalid(#[values(1, 5)] size: usize) {
        let space = IndexSpace::new(size);
        testing::check_from_index_invalid(&space);
    }
}
