//! `IndexSpace` definition
use super::{CategoricalSpace, FiniteSpace, Space};
use rand::distributions::Distribution;
use rand::Rng;
use std::fmt;

/// An index space; integers 0 .. size-1
#[derive(Debug, Clone)]
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
