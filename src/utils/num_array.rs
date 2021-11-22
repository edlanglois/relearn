//! Numeric array interfaces.

use super::tensor::UniqueTensor;
use ndarray::{Array1, Array2, ArrayView, ArrayViewMut, Ix1, Ix2};
use num_traits::{One, Zero};
use tch::{kind::Element, Tensor};

/// A one-dimensional numeric array.
pub trait NumArray1D {
    type Elem;

    /// Create a new instance of the given size with all elements initialized to zero.
    fn zeros(size: usize) -> Self;

    /// Create a new instance of the given size with all elements initialized to one.
    fn ones(size: usize) -> Self;

    /// View as a slice.
    fn as_slice(&self) -> &[Self::Elem];

    /// View as a mutable slice.
    fn as_slice_mut(&mut self) -> &mut [Self::Elem];
}

/// A two-dimensional numeric array.
pub trait NumArray2D {
    type Elem;

    /// Create a new instance of the given size with all elements initialized to zero.
    fn zeros(size: (usize, usize)) -> Self;

    /// Create a new instance of the given size with all elements initialized to one.
    fn ones(size: (usize, usize)) -> Self;

    /// View as a two-dimensional [`ArrayView`].
    fn view(&self) -> ArrayView<Self::Elem, Ix2>;

    /// Mutable view as a two-dimensional [`ArrayViewMut`].
    fn view_mut(&mut self) -> ArrayViewMut<Self::Elem, Ix2>;
}

/// Build by converting from an associated [`NumArray1D`].
pub trait BuildFromArray1D: From<Self::Array> {
    type Array: NumArray1D;
}
impl<T: NumArray1D> BuildFromArray1D for T {
    type Array = Self;
}

/// Build by converting from an associated [`NumArray2D`].
pub trait BuildFromArray2D: From<Self::Array> {
    type Array: NumArray2D;
}
impl<T: NumArray2D> BuildFromArray2D for T {
    type Array = Self;
}

impl<A: Clone + Zero + One> NumArray1D for Array1<A> {
    type Elem = A;
    #[inline(always)]
    fn zeros(size: usize) -> Self {
        Self::zeros(size)
    }
    #[inline(always)]
    fn ones(size: usize) -> Self {
        Self::ones(size)
    }
    #[inline(always)]
    fn as_slice(&self) -> &[Self::Elem] {
        self.as_slice().unwrap()
    }
    #[inline(always)]
    fn as_slice_mut(&mut self) -> &mut [Self::Elem] {
        self.as_slice_mut().unwrap()
    }
}

impl<A: Clone + Zero + One> NumArray2D for Array2<A> {
    type Elem = A;
    #[inline(always)]
    fn zeros(size: (usize, usize)) -> Self {
        Self::zeros(size)
    }
    #[inline(always)]
    fn ones(size: (usize, usize)) -> Self {
        Self::ones(size)
    }
    #[inline(always)]
    fn view(&self) -> ArrayView<Self::Elem, Ix2> {
        self.view()
    }
    #[inline(always)]
    fn view_mut(&mut self) -> ArrayViewMut<Self::Elem, Ix2> {
        self.view_mut()
    }
}

impl<A: Element> NumArray1D for UniqueTensor<A, Ix1> {
    type Elem = A;

    #[inline(always)]
    fn zeros(size: usize) -> Self {
        Self::zeros(size)
    }
    #[inline(always)]
    fn ones(size: usize) -> Self {
        Self::ones(size)
    }
    #[inline(always)]
    fn as_slice(&self) -> &[Self::Elem] {
        self.as_slice()
    }
    #[inline(always)]
    fn as_slice_mut(&mut self) -> &mut [Self::Elem] {
        self.as_slice_mut()
    }
}

impl<A: Element> NumArray2D for UniqueTensor<A, Ix2> {
    type Elem = A;

    #[inline(always)]
    fn zeros(size: (usize, usize)) -> Self {
        Self::zeros(size)
    }
    #[inline(always)]
    fn ones(size: (usize, usize)) -> Self {
        Self::ones(size)
    }
    #[inline(always)]
    fn view(&self) -> ArrayView<Self::Elem, Ix2> {
        self.array_view()
    }
    #[inline(always)]
    fn view_mut(&mut self) -> ArrayViewMut<Self::Elem, Ix2> {
        self.array_view_mut()
    }
}

impl BuildFromArray1D for Tensor {
    type Array = UniqueTensor<f32, Ix1>;
}
impl BuildFromArray2D for Tensor {
    type Array = UniqueTensor<f32, Ix2>;
}
