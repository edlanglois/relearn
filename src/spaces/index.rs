use super::{FiniteSpace, Space};
use rand::distributions::Distribution;
use rand::Rng;

/// An index space; integers 0 .. size-1
pub struct IndexSpace {
    pub size: usize,
}

impl IndexSpace {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Space for IndexSpace {
    type Element = usize;

    fn contains(&self, value: &usize) -> bool {
        value < &self.size
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
