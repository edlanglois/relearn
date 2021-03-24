use super::Space;

/// A space containing finitely many elements.
pub trait FiniteSpace: Space {
    /// The number of elements in the space.
    fn size(&self) -> usize;

    /// Get the index of an element.
    fn to_index(&self, element: &Self::Element) -> usize;

    /// Try to convert an index to an element.
    ///
    /// If None is returned then the index was invalid.
    /// It is allowed that Some value may be returned even if the index is invalid.
    /// If you need to validate the returned value, use contains().
    fn from_index(&self, index: usize) -> Option<Self::Element>;
}

#[cfg(test)]
pub mod finite_space_checks {
    use super::FiniteSpace;
    use rand::prelude::*;

    /// Check paired from_index and to_index for each i in 0..space.size()
    pub fn check_from_to_index_iter_size<T: FiniteSpace>(space: T) {
        for index in 0..space.size() {
            let element = space.from_index(index).unwrap();
            assert!(space.contains(&element));
            let index2 = space.to_index(&element);
            assert_eq!(index2, index);
        }
    }

    // /// Check paired from_index and to_index for random indices in 0..space.size()
    // pub fn check_from_to_index_random<T: FiniteSpace>(space: T, num_samples: u32) {
    //     let size = space.size();
    //     let mut rng = StdRng::seed_from_u64(2);
    //     for _ in 0..num_samples {
    //         let index = rng.gen_range(0..size);
    //         let element = space.from_index(index).unwrap();
    //         assert!(space.contains(&element));
    //         let index2 = space.to_index(&element);
    //         assert_eq!(index2, index);
    //     }
    // }

    /// Check from_index for element sampled from the space.
    pub fn check_from_index_sampled<T: FiniteSpace>(space: T, num_samples: u32) {
        let mut rng = StdRng::seed_from_u64(3);
        let size = space.size();
        for _ in 0..num_samples {
            let element = space.sample(&mut rng);
            let index = space.to_index(&element);
            assert!(index < size);
        }
    }
}
