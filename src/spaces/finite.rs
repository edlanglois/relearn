use super::Space;

/// A space containing finitely many elements.
pub trait FiniteSpace: Space {
    /// The number of elements in the space.
    fn len(&self) -> usize;

    /// The element at a particular index
    fn index(&self, index: usize) -> Self::Element;

    /// The index of an element in the space.
    fn index_of(&self, element: &Self::Element) -> usize;
}
