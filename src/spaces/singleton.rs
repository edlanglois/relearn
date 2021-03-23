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

    fn as_loggable(&self, _value: &Self::Element) -> Loggable {
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
