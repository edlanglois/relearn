use super::{BaseSpace, FiniteSpace, Space};
use crate::loggers::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
use std::fmt;

/// An index space; integers 0 .. size-1
#[derive(Debug, Clone)]
pub struct IndexSpace {
    pub size: usize,
}

impl IndexSpace {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl fmt::Display for IndexSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IndexSpace({})", self.size)
    }
}

impl BaseSpace for IndexSpace {}

impl Space for IndexSpace {
    type Element = usize;

    fn contains(&self, value: &usize) -> bool {
        value < &self.size
    }

    fn as_loggable(&self, value: &usize) -> Loggable {
        Loggable::IndexSample {
            value: *value,
            size: self.size,
        }
    }
}

impl FiniteSpace for IndexSpace {
    fn len(&self) -> usize {
        self.size
    }

    fn index(&self, index: usize) -> usize {
        index
    }

    fn index_of(&self, element: &usize) -> usize {
        *element
    }
}

impl Distribution<usize> for IndexSpace {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        rng.gen_range(0..self.size)
    }
}
