mod finite;
mod index;
mod indexed_type;
mod singleton;

pub use finite::FiniteSpace;
pub use index::IndexSpace;
pub use indexed_type::{Indexed, IndexedTypeSpace};
pub use singleton::SingletonSpace;

use crate::logging::Loggable;
use rand::distributions::Distribution;

/// A mathematical space
pub trait Space: Distribution<<Self as Space>::Element> {
    type Element;

    /// Check if the space contains a particular value
    fn contains(&self, value: &Self::Element) -> bool;

    /// Convert an element into a loggable object
    fn as_loggable(&self, element: &Self::Element) -> Loggable;
}

#[cfg(test)]
pub mod space_checks {
    use super::Space;
    use rand::prelude::*;

    pub fn check_contains_samples<T: Space>(space: T, num_samples: u32) {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..num_samples {
            let element = space.sample(&mut rng);
            assert!(space.contains(&element));
        }
    }
}
