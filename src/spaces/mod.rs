mod finite;
mod index;
mod singleton;

pub use finite::FiniteSpace;
pub use index::IndexSpace;
pub use singleton::SingletonSpace;

use crate::logging::Loggable;
use rand::distributions::Distribution;
use std::fmt::{Debug, Display};

/// A mathematical space
pub trait Space: BaseSpace + Distribution<<Self as Space>::Element> {
    type Element;

    /// Check if the space contains a particular value
    fn contains(&self, value: &Self::Element) -> bool;

    /// Convert an element into a loggable object
    fn as_loggable(&self, value: &Self::Element) -> Loggable;
}

/// An object-safe base description of a mathematical space.
pub trait BaseSpace: Display + Debug {}
