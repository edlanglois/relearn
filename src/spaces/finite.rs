//! FiniteSpace trait definition
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
