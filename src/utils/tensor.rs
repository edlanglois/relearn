use ndarray::{Array, ArrayView, ArrayViewMut, Dim, Dimension, IntoDimension, Ix, IxDyn};
use std::marker::PhantomData;
use tch::{kind::Element, Device, Kind, Tensor};

/// A unique tensor object.
///
/// Given an ordinary [`Tensor`], it is impossible to reason about the lifetime of the data at
/// [`Tensor::data_ptr`]. Copies created by [`Tensor::shallow_clone`] share the same underlying
/// tensor object and can cause the data memory to be moved or reallocated at any time (for
/// example, by calling `Tensor::resize_`].
///
/// To avoid this issue `UniqueTensor` manages the creation of the tensor such that it has
/// exclusive access to the underlying data.
///
/// The managed tensor always lives on the CPU.
#[derive(Debug)]
pub struct UniqueTensor<E, D>
where
    D: Dimension,
{
    tensor: Tensor,
    /// Track element type to avoid runtime checks
    element_type: PhantomData<*const E>,
    /// Track shape to avoid runtime checks
    shape: D,
    /// An empty array to view if `tensor` is empty.
    ///
    /// Necessary because `tensor.data_ptr()` is invalid (null) when `tensor` has no elements.
    empty_fallback: Option<Array<E, D>>,
}

fn empty_array<A, D: Dimension>(shape: D) -> Array<A, D> {
    Array::from_shape_simple_fn(shape, || panic!("array is not empty"))
}

impl<E, D> UniqueTensor<E, D>
where
    E: Element,
    D: Dimension + IntoTorchShape,
{
    /// Create a zero-initialized tensor.
    pub fn zeros<Sh: IntoDimension<Dim = D>>(shape: Sh) -> Self {
        unsafe {
            Self::from_tensor_fn(shape, |shape, kind| {
                Tensor::zeros(shape, (kind, Device::Cpu))
            })
        }
    }

    /// Create a one-initialized tensor.
    pub fn ones<Sh: IntoDimension<Dim = D>>(shape: Sh) -> Self {
        unsafe {
            Self::from_tensor_fn(shape, |shape, kind| {
                Tensor::ones(shape, (kind, Device::Cpu))
            })
        }
    }

    /// Initialize given a tensor construction function.
    ///
    /// # Safety
    /// The construct tensor must
    ///     * have number of elements corresponding to `shape`,
    ///     * have elements of type `E`,
    ///     * use `Device::Cpu`, and
    ///     * uniquely manage its own memory (e.g. no `shallow_clone`).
    ///
    /// # Panics
    /// If the total number of elements exceeds `isize::MAX`.
    unsafe fn from_tensor_fn<Sh, F>(shape: Sh, f: F) -> Self
    where
        Sh: IntoDimension<Dim = D>,
        F: FnOnce(&[i64], Kind) -> Tensor,
    {
        let shape = shape.into_dimension();
        match shape.size_checked() {
            Some(size) if size < isize::MAX as usize => {}
            _ => panic!("size must not exceed isize::MAX"),
        }
        let empty_fallback = if shape.size() == 0 {
            Some(empty_array(shape.clone()))
        } else {
            None
        };
        Self {
            tensor: f(shape.clone().into_torch_shape().as_ref(), E::KIND),
            element_type: PhantomData,
            shape,
            empty_fallback,
        }
    }
}

impl<E, D> UniqueTensor<E, D>
where
    E: Element,
    D: Dimension,
{
    pub fn array_view(&self) -> ArrayView<E, D> {
        // # Safety
        //
        // ✓ **Elements must live as long as 'a (in ArrayView<'a, E, D>).**
        //   UniqueTensor has exclusive access to the data managed by the tensor and only allows
        //   access that is consistent with the lifetime of self.
        //   The data memory cannot be modified without a &mut self.
        //
        // ✓ **ptr must be non-null and aligned, and it must be safe to .offset() ptr by zero.**
        //   This is up to torch but it should be true for non-empty tensors since data is being
        //   stored at this pointer value. In the case of empty tensors, the data pointer is null
        //   so we view the empty fallback array instead.
        //
        // ? **It must be safe to .offset() the pointer repeatedly along all axes and calculate the
        //   counts for the .offset() calls without overflow, even if the array is empty or the
        //   elements are zero-sized.**
        //   Up to pytorch but again it should be true since the full tensor's worth of data is
        //   being stored at this pointer value.
        //
        // ✓ **The product of non-zero axis lengths must not exceed isize::MAX.**
        //   Asserted in constructors; but probably a similar constraint applies to the tensor
        //   creation by pytorch.
        //
        // ✓ **Strides must be non-negative.**
        //   Dimension as IntoDimension as Into<StrideShape> always uses C-style strides
        //   which have a value of 0 or 1 depending on the array shape.

        if let Some(ref empty_fallback) = self.empty_fallback {
            assert_eq!(self.shape.size(), 0);
            empty_fallback.view()
        } else {
            unsafe { ArrayView::from_shape_ptr(self.shape.clone(), self.tensor.data_ptr() as _) }
        }
    }

    pub fn array_view_mut(&mut self) -> ArrayViewMut<E, D> {
        // # Safety
        // See `Self::array_view` implementation
        if let Some(ref mut empty_fallback) = self.empty_fallback {
            assert_eq!(self.shape.size(), 0);
            empty_fallback.view_mut()
        } else {
            unsafe { ArrayViewMut::from_shape_ptr(self.shape.clone(), self.tensor.data_ptr() as _) }
        }
    }
}

impl<E, D: Dimension> From<UniqueTensor<E, D>> for Tensor {
    fn from(unique: UniqueTensor<E, D>) -> Self {
        unique.tensor
    }
}

impl<'a, E, D> From<&'a UniqueTensor<E, D>> for ArrayView<'a, E, D>
where
    E: Element,
    D: Dimension,
{
    fn from(unique: &'a UniqueTensor<E, D>) -> Self {
        unique.array_view()
    }
}

fn to_i64(x: Ix) -> i64 {
    x.try_into().expect("dimension too large")
}

/// Convert an ndarray-style dimension into the shape type used by [`tch`].
pub trait IntoTorchShape {
    type TorchDim: AsRef<[i64]>;
    fn into_torch_shape(self) -> Self::TorchDim;
}
impl IntoTorchShape for IxDyn {
    type TorchDim = Vec<i64>;
    fn into_torch_shape(self) -> Self::TorchDim {
        self.as_array_view()
            .into_iter()
            .map(|&x| to_i64(x))
            .collect()
    }
}
impl IntoTorchShape for Dim<[Ix; 0]> {
    type TorchDim = [i64; 0];
    fn into_torch_shape(self) -> Self::TorchDim {
        []
    }
}
impl IntoTorchShape for Dim<[Ix; 1]> {
    type TorchDim = [i64; 1];
    fn into_torch_shape(self) -> Self::TorchDim {
        [self.into_pattern() as _]
    }
}
impl IntoTorchShape for Dim<[Ix; 2]> {
    type TorchDim = [i64; 2];
    fn into_torch_shape(self) -> Self::TorchDim {
        let (a, b) = self.into_pattern();
        [to_i64(a), to_i64(b)]
    }
}
impl IntoTorchShape for Dim<[Ix; 3]> {
    type TorchDim = [i64; 3];
    fn into_torch_shape(self) -> Self::TorchDim {
        let (a, b, c) = self.into_pattern();
        [to_i64(a), to_i64(b), to_i64(c)]
    }
}
impl IntoTorchShape for Dim<[Ix; 4]> {
    type TorchDim = [i64; 4];
    fn into_torch_shape(self) -> Self::TorchDim {
        let (a, b, c, d) = self.into_pattern();
        [to_i64(a), to_i64(b), to_i64(c), to_i64(d)]
    }
}
impl IntoTorchShape for Dim<[Ix; 5]> {
    type TorchDim = [i64; 5];
    #[allow(clippy::many_single_char_names)]
    fn into_torch_shape(self) -> Self::TorchDim {
        let (a, b, c, d, e) = self.into_pattern();
        [to_i64(a), to_i64(b), to_i64(c), to_i64(d), to_i64(e)]
    }
}
impl IntoTorchShape for Dim<[Ix; 6]> {
    type TorchDim = [i64; 6];
    #[allow(clippy::many_single_char_names)]
    fn into_torch_shape(self) -> Self::TorchDim {
        let (a, b, c, d, e, f) = self.into_pattern();
        [
            to_i64(a),
            to_i64(b),
            to_i64(c),
            to_i64(d),
            to_i64(e),
            to_i64(f),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn zeros() {
        let u = UniqueTensor::<f32, _>::zeros([2, 4, 3]);
        let tensor: Tensor = u.into();
        assert_eq!(tensor.size(), vec![2, 4, 3]);
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(tensor.device(), Device::Cpu);
        assert_eq!(
            tensor,
            Tensor::zeros(&[2, 4, 3], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn ones() {
        let u = UniqueTensor::<f32, _>::ones([2, 4, 3]);
        let tensor: Tensor = u.into();
        assert_eq!(tensor.size(), vec![2, 4, 3]);
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(tensor.device(), Device::Cpu);
        assert_eq!(tensor, Tensor::ones(&[2, 4, 3], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn test_array_view_f32() {
        let u = UniqueTensor::<f32, _>::ones([2, 4, 3]);
        let view = u.array_view();
        assert_eq!(view.dim(), (2, 4, 3));
        assert_eq!(view, Array::ones((2, 4, 3)));
    }

    #[test]
    #[allow(clippy::unit_cmp)]
    fn test_array_view_i64_scalar() {
        let u = UniqueTensor::<i64, _>::ones([]);
        let view = u.array_view();
        assert_eq!(view.dim(), ());
        assert_eq!(view.into_scalar(), &1);
    }

    #[test]
    fn test_array_view_f32_empty() {
        let u = UniqueTensor::<f32, _>::ones([0]);
        let view = u.array_view();
        assert_eq!(view.dim(), 0);
        assert!(view.as_slice().unwrap().is_empty());
    }

    #[test]
    fn test_array_view_mut() {
        let mut u = UniqueTensor::<i32, _>::ones([3, 4]);
        let mut view = u.array_view_mut();
        for (i, mut row) in view.rows_mut().into_iter().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (i * 10 + j).try_into().unwrap();
            }
        }
        let expected = arr2(&[[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]);
        assert_eq!(view, expected); // Compare as arrays
        let t: Tensor = u.into();
        let expected: Tensor = expected.try_into().unwrap();
        assert_eq!(t, expected); // Compare as tensors
    }

    #[test]
    fn test_array_view_mut_empty() {
        let mut u = UniqueTensor::<f32, _>::ones([2, 0, 3]);
        let mut view = u.array_view_mut();
        assert!(view.as_slice_mut().unwrap().is_empty());
    }
}
