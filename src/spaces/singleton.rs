use super::{FiniteSpace, Space};
use crate::logging::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
use std::fmt;

/// A space containing a single element.
#[derive(Debug, Clone)]
pub struct SingletonSpace {}

impl SingletonSpace {
    pub fn new() -> Self {
        SingletonSpace {}
    }
}

impl fmt::Display for SingletonSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SingletonSpace")
    }
}

impl Space for SingletonSpace {
    type Element = ();

    fn contains(&self, _value: &Self::Element) -> bool {
        true
    }

    fn as_loggable(&self, _element: &Self::Element) -> Loggable {
        Loggable::Nothing
    }
}

impl FiniteSpace for SingletonSpace {
    fn size(&self) -> usize {
        1
    }

    fn to_index(&self, _element: &Self::Element) -> usize {
        0
    }

    fn from_index(&self, _index: usize) -> Option<Self::Element> {
        Some(())
    }
}

impl Distribution<<Self as Space>::Element> for SingletonSpace {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> <Self as Space>::Element {
        ()
    }
}

#[cfg(test)]
mod tests {
    use super::super::finite::finite_space_checks;
    use super::super::space_checks;
    use super::*;

    #[test]
    fn contains_samples() {
        let space = SingletonSpace::new();
        space_checks::check_contains_samples(space, 10);
    }

    #[test]
    fn from_to_index_iter_size() {
        let space = SingletonSpace::new();
        finite_space_checks::check_from_to_index_iter_size(space);
    }

    #[test]
    fn from_index_sampled() {
        let space = SingletonSpace::new();
        finite_space_checks::check_from_index_sampled(space, 10);
    }
}
