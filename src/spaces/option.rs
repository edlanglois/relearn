//! Optional space definition.
use super::{ElementRefInto, FiniteSpace, Space};
use crate::logging::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
use std::fmt;

/// A space whose elements are either `None` or `Some(inner_elem)`.
#[derive(Debug, Clone)]
pub struct OptionSpace<S> {
    pub inner: S,
}

impl<S> OptionSpace<S> {
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: fmt::Display> fmt::Display for OptionSpace<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OptionSpace<{}>", self.inner)
    }
}

impl<S: Space> Space for OptionSpace<S> {
    type Element = Option<<S as Space>::Element>;

    fn contains(&self, value: &Self::Element) -> bool {
        match value {
            None => true,
            Some(inner_value) => self.inner.contains(inner_value),
        }
    }
}

impl<S: FiniteSpace> FiniteSpace for OptionSpace<S> {
    fn size(&self) -> usize {
        1 + self.inner.size()
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        match element {
            None => 0,
            Some(inner_elem) => 1 + self.inner.to_index(inner_elem),
        }
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        if index == 0 {
            Some(None)
        } else {
            Some(Some(self.inner.from_index(index - 1)?))
        }
    }
}

impl<S> Distribution<<Self as Space>::Element> for OptionSpace<S>
where
    S: Space + Distribution<<S as Space>::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        // Sample None half of the time.
        if rng.gen() {
            None
        } else {
            Some(self.inner.sample(rng))
        }
    }
}

// NOTE: This should maybe be the equivalent of TryInto instead of Into
impl<S: Space> ElementRefInto<Loggable> for OptionSpace<S> {
    fn elem_ref_into(&self, _element: &Self::Element) -> Loggable {
        // No clear way to convert structured elements into Loggable
        Loggable::Nothing
    }
}

#[cfg(test)]
mod option_space {
    use super::super::{testing, IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn contains_none() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert!(space.contains(&None));
    }

    #[test]
    fn contains_some() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert!(space.contains(&Some(())));
    }

    #[test]
    fn contains_samples_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_contains_samples(&space, 100);
    }

    #[test]
    fn contains_samples_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_contains_samples(&space, 100);
    }

    #[test]
    fn from_to_index_iter_size_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_iter_size_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_index_invalid_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn from_index_invalid_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_index_invalid(&space);
    }
}
