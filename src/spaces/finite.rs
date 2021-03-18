use super::Space;

/// A space containing finitely many elements.
pub trait FiniteSpace: Space {
    /// The number of elements in the space.
    fn len(&self) -> usize;

    /// The element at a particular index
    ///
    /// If None is returned then the index was invalid.
    /// It is allowed that Some value may be returned even if the index is invalid.
    /// If you need to validate the returned value, use contains().
    fn index(&self, index: usize) -> Option<Self::Element>;

    /// The index of an element in the space.
    fn index_of(&self, element: &Self::Element) -> usize;
}
