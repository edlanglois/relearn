//! A generic array interface.
use ndarray::{Array, ArrayBase, DataMut, Dim, Dimension, Ix, RawData};
use num_traits::Zero;
use std::convert::TryInto;
use tch::{Device, Tensor};

/// A basic multidimensional array with simple operations.
pub trait BasicArray<T, const N: usize> {
    /// Allocate a new array with the given shape.
    ///
    /// Also returns a boolean indiciating whether the array is zero-initialized (`true`)
    /// or contains arbitrary values (`false`).
    fn allocate(shape: [usize; N]) -> (Self, bool)
    where
        Self: Sized,
    {
        (Self::zeros(shape), true)
    }

    /// Create a zero-initialized array with the given shape.
    fn zeros(shape: [usize; N]) -> Self;
}

/// A basic mutable multidimensional array with simple operations.
pub trait BasicArrayMut {
    /// Zero out the array in-place.
    fn zero_(&mut self);
}

impl<T: tch::kind::Element, const N: usize> BasicArray<T, N> for Tensor {
    fn allocate(shape: [usize; N]) -> (Self, bool)
    where
        Self: Sized,
    {
        let shape: Vec<i64> = shape.iter().map(|&x| x.try_into().unwrap()).collect();
        (Self::empty(&shape, (T::KIND, Device::Cpu)), false)
    }

    fn zeros(shape: [usize; N]) -> Self {
        let shape: Vec<i64> = shape.iter().map(|&x| x.try_into().unwrap()).collect();
        Self::zeros(&shape, (T::KIND, Device::Cpu))
    }
}

impl BasicArrayMut for Tensor {
    fn zero_(&mut self) {
        let _ = self.zero_();
    }
}

macro_rules! basic_ndarray {
    ($n:expr) => {
        impl<T> BasicArray<T, $n> for Array<T, Dim<[Ix; $n]>>
        where
            T: Clone + Zero,
        {
            fn zeros(shape: [usize; $n]) -> Self {
                Self::zeros(shape)
            }
        }
    };
}

basic_ndarray!(1);
basic_ndarray!(2);
basic_ndarray!(3);
basic_ndarray!(4);
basic_ndarray!(5);
basic_ndarray!(6);

impl<T, D> BasicArrayMut for ArrayBase<T, D>
where
    T: DataMut,
    <T as RawData>::Elem: Clone + Zero,
    D: Dimension,
{
    fn zero_(&mut self) {
        self.fill(Zero::zero());
    }
}
