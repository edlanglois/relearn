//! Space test utilities
use super::{FiniteSpace, SampleSpace};
use rand::prelude::*;

/// Check that space contains samples it generates
pub fn check_contains_samples<T: SampleSpace>(space: T, num_samples: u32) {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..num_samples {
        let element = space.sample(&mut rng);
        assert!(space.contains(&element));
    }
}

/// Check paired from_index and to_index for each i in 0..space.size()
pub fn check_from_to_index_iter_size<T: FiniteSpace>(space: T) {
    for index in 0..space.size() {
        let element = space.from_index(index).unwrap();
        assert!(space.contains(&element));
        let index2 = space.to_index(&element);
        assert_eq!(index2, index);
    }
}

/// Check paired from_index and to_index for random indices in 0..space.size()
pub fn check_from_to_index_random<T: FiniteSpace>(space: T, num_samples: u32) {
    let size = space.size();
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..num_samples {
        let index = rng.gen_range(0, size);
        let element = space.from_index(index).unwrap();
        assert!(space.contains(&element));
        let index2 = space.to_index(&element);
        assert_eq!(index2, index);
    }
}

/// Check from_index for element sampled from the space.
pub fn check_from_index_sampled<T: FiniteSpace + SampleSpace>(space: T, num_samples: u32) {
    let mut rng = StdRng::seed_from_u64(3);
    let size = space.size();
    for _ in 0..num_samples {
        let element = space.sample(&mut rng);
        let index = space.to_index(&element);
        assert!(index < size);
    }
}
