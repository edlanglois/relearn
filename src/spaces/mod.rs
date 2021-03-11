pub mod finite;
pub mod index;

pub use finite::FiniteSpace;
pub use index::IndexSpace;

use rand::distributions::Distribution;

/// A mathematical space
pub trait Space: Distribution<<Self as Space>::Element> {
    type Element;

    /// Check if the space contains a particular value
    fn contains(&self, value: &Self::Element) -> bool;
}
