//! Option space definition.
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, ElementRefInto, FeatureSpace,
    FeatureSpaceOut, FiniteSpace, Space,
};
use crate::logging::Loggable;
use ndarray::{s, Array, ArrayBase, ArrayViewMut, DataMut, Ix1, Ix2};
use num_traits::{One, Zero};
use rand::distributions::Distribution;
use rand::Rng;
use std::convert::TryInto;
use std::fmt;
use std::marker::PhantomData;
use tch::{Device, Kind, Tensor};

/// A space whose elements are either `None` or `Some(inner_elem)`.
///
/// The feature vectors are
/// * `1, 0, ..., 0` for `None`
/// * `0, inner_feature_vector(x)` for `Some(x)`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OptionSpace<S> {
    pub inner: S,
}

impl<S> OptionSpace<S> {
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: fmt::Display> fmt::Display for OptionSpace<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OptionSpace<{}>", self.inner)
    }
}

impl<S: Space> Space for OptionSpace<S> {
    type Element = Option<S::Element>;

    fn contains(&self, value: &Self::Element) -> bool {
        match value {
            None => true,
            Some(inner_value) => self.inner.contains(inner_value),
        }
    }
}

impl<S: FiniteSpace> FiniteSpace for OptionSpace<S> {
    fn size(&self) -> usize {
        1 + self.inner.size()
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        match element {
            None => 0,
            Some(inner_elem) => 1 + self.inner.to_index(inner_elem),
        }
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        if index == 0 {
            Some(None)
        } else {
            Some(Some(self.inner.from_index(index - 1)?))
        }
    }
}

impl<S: BaseFeatureSpace> BaseFeatureSpace for OptionSpace<S> {
    fn num_features(&self) -> usize {
        1 + self.inner.num_features()
    }
}

// Feature vectors are:
// * `1, 0, ..., 0` for `None`
// * `0, feature_vector(x)` for `Some(x)`.
impl<S, T> FeatureSpace<Array<T, Ix1>> for OptionSpace<S>
where
    S: for<'a> FeatureSpaceOut<ArrayViewMut<'a, T, Ix1>>,
    T: Clone + Zero + One,
{
    fn features(&self, element: &Self::Element) -> Array<T, Ix1> {
        let mut out = Array::zeros(self.num_features());
        self.features_out(element, &mut out, true);
        out
    }
}

impl<S, T> FeatureSpaceOut<ArrayBase<T, Ix1>> for OptionSpace<S>
where
    S: for<'a> FeatureSpaceOut<ArrayViewMut<'a, T::Elem, Ix1>>,
    T: DataMut,
    T::Elem: Clone + Zero + One,
{
    fn features_out(&self, element: &Self::Element, out: &mut ArrayBase<T, Ix1>, zeroed: bool) {
        if let Some(inner_elem) = element {
            self.inner
                .features_out(inner_elem, &mut out.slice_mut(s![1..]), zeroed);
            if !zeroed {
                out[0] = Zero::zero()
            }
        } else {
            if !zeroed {
                out.slice_mut(s![1..]).fill(Zero::zero());
            }
            out[0] = One::one();
        }
    }
}

impl<S, T> PhantomBatchFeatureSpace<Array<T, Ix2>> for OptionSpace<S>
where
    S: for<'a> FeatureSpaceOut<ArrayViewMut<'a, T, Ix1>>,
    T: Clone + Zero + One,
{
    fn phantom_batch_features<'a, I>(
        &self,
        elements: I,
        marker: PhantomData<&'a Self::Element>,
    ) -> Array<T, Ix2>
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
    {
        let elements = elements.into_iter();
        let mut out = Array::zeros([elements.len(), self.num_features()]);
        self.phantom_batch_features_out(elements, &mut out, true, marker);
        out
    }
}

impl<S, T> PhantomBatchFeatureSpaceOut<ArrayBase<T, Ix2>> for OptionSpace<S>
where
    S: for<'a> FeatureSpaceOut<ArrayViewMut<'a, T::Elem, Ix1>>,
    T: DataMut,
    T::Elem: Clone + Zero + One,
{
    fn phantom_batch_features_out<'a, I>(
        &self,
        elements: I,
        out: &mut ArrayBase<T, Ix2>,
        zeroed: bool,
        _marker: PhantomData<&'a Self::Element>,
    ) where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        for (mut row, element) in out.outer_iter_mut().zip(elements) {
            self.features_out(element, &mut row, zeroed);
        }
    }
}

impl<S: FeatureSpaceOut<Tensor>> FeatureSpace<Tensor> for OptionSpace<S> {
    fn features(&self, element: &Self::Element) -> Tensor {
        let mut out = Tensor::empty(&[self.num_features() as i64], (Kind::Float, Device::Cpu));
        self.features_out(element, &mut out, false);
        out
    }
}

impl<S: FeatureSpaceOut<Tensor>> FeatureSpaceOut<Tensor> for OptionSpace<S> {
    fn features_out(&self, element: &Self::Element, out: &mut Tensor, zeroed: bool) {
        let rest_size = self.inner.num_features();
        let [mut first, mut rest]: [Tensor; 2] = out
            .split_with_sizes(&[1, rest_size as i64], -1)
            .try_into()
            .unwrap();
        if let Some(inner_elem) = element {
            if !zeroed {
                let _ = first.zero_();
            }
            self.inner.features_out(inner_elem, &mut rest, zeroed);
        } else {
            let _ = first.fill_(1.0);
            if !zeroed {
                let _ = rest.zero_();
            }
        }
    }
}

impl<S> PhantomBatchFeatureSpace<Tensor> for OptionSpace<S>
where
    S: for<'a> FeatureSpaceOut<ArrayViewMut<'a, f32, Ix1>>,
{
    fn phantom_batch_features<'a, I>(
        &self,
        elements: I,
        _marker: PhantomData<&'a Self::Element>,
    ) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
    {
        // NOTE:
        // Constructing an array of features then copying to a tensor is nearly as fast as the
        // fastest direct-to-torch implementation (using a more complex batch_features interface)
        // while being considerably simpler and less error-prone.
        // It is very difficult construct torch tensors with complex inner data layouts
        // and many methods are orders of magnitude slower than constructing into an Array
        // and copying.
        let features: Array<f32, _> = self.batch_features(elements);
        features.try_into().unwrap()
    }
}

impl<S: BatchFeatureSpace<Tensor>> PhantomBatchFeatureSpaceOut<Tensor> for OptionSpace<S> {
    fn phantom_batch_features_out<'a, I>(
        &self,
        elements: I,
        out: &mut Tensor,
        zeroed: bool,
        _: PhantomData<&'a Self::Element>,
    ) where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        let mut none_indices = Vec::new();
        let mut some_elements = Vec::new();
        let mut some_indices = Vec::new();
        for (i, element) in elements.into_iter().enumerate() {
            if let Some(x) = element {
                some_elements.push(x);
                some_indices.push(i as i64);
            } else {
                none_indices.push(i as i64);
            }
        }
        let rest_size = self.inner.num_features();
        let [mut first, mut rest]: [Tensor; 2] = out
            .split_with_sizes(&[1, rest_size as i64], -1)
            .try_into()
            .unwrap();

        if !zeroed {
            let _ = out.zero_();
        }
        let _ = first.index_fill_(-2, &Tensor::of_slice(&none_indices), 1.0);

        // As far as I can tell, you can't make a view that irregularly includes just some rows.
        // Acting row-by-row is extremely slow for torch tensors
        // So create a new tensor containg just the inner features densely packed,
        // then copy rows into the output tensor at the correct spots.
        let some_features = self.inner.batch_features(some_elements);
        let _ = rest.index_copy_(-2, &Tensor::of_slice(&some_indices), &some_features);
    }
}

impl<S> Distribution<<Self as Space>::Element> for OptionSpace<S>
where
    S: Space + Distribution<S::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        // Sample None half of the time.
        if rng.gen() {
            None
        } else {
            Some(self.inner.sample(rng))
        }
    }
}

// NOTE: ElementRefTryInto instead of ElementRefInto?
impl<S: Space> ElementRefInto<Loggable> for OptionSpace<S> {
    fn elem_ref_into(&self, _element: &Self::Element) -> Loggable {
        // No clear way to convert structured elements into Loggable
        Loggable::Nothing
    }
}

/// Hack to allow implementing [`FeatureSpace`] for [`OptionSpace`].
///
/// Uses an alternative definition of `batch_features` that takes `PhantomData`,
/// which apparently helps the compiler reason about lifetimes.
///
/// Then in the `BatchFeatureSpace` implementation we use `Self: PhantomBatchFeatureSpace`
/// which hides the fact that `Self::Element` is an `Option`.
/// The compiler accepts lifetime bounds for an arbitrary `S::Element`
/// but not for `Option<S::Element>`...
///
/// # References
/// * <https://users.rust-lang.org/t/lifetime/59967>
/// * <https://github.com/rust-lang/rust/issues/85451>
pub trait PhantomBatchFeatureSpace<T2>: Space {
    fn phantom_batch_features<'a, I>(
        &self,
        elements: I,
        _marker: PhantomData<&'a Self::Element>,
    ) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a;
}

/// Hack to allow implementing [`BatchFeatureSpaceOut`] for [`OptionSpace`].
///
/// See [`PhantomBatchFeatureSpace`] for more details.
pub trait PhantomBatchFeatureSpaceOut<T2>: Space {
    fn phantom_batch_features_out<'a, I>(
        &self,
        elements: I,
        out: &mut T2,
        zeroed: bool,
        _marker: PhantomData<&'a Self::Element>,
    ) where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a;
}

/// Feature vectors are:
/// * `1, 0, ..., 0` for `None`
/// * `0, feature_vector(x)` for `Some(x)`.
impl<S, T2> BatchFeatureSpace<T2> for OptionSpace<S>
where
    S: BaseFeatureSpace,
    Self: PhantomBatchFeatureSpace<T2>,
{
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
    {
        self.phantom_batch_features(elements, PhantomData)
    }
}

impl<S, T2> BatchFeatureSpaceOut<T2> for OptionSpace<S>
where
    S: BaseFeatureSpace,
    Self: PhantomBatchFeatureSpaceOut<T2>,
{
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut T2, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        self.phantom_batch_features_out(elements, out, zeroed, PhantomData)
    }
}

#[cfg(test)]
mod space {
    use super::super::{testing, IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn contains_none() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert!(space.contains(&None));
    }

    #[test]
    fn contains_some() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert!(space.contains(&Some(())));
    }

    #[test]
    fn contains_samples_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_contains_samples(&space, 100);
    }

    #[test]
    fn contains_samples_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_contains_samples(&space, 100);
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::{testing, IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn from_to_index_iter_size_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_iter_size_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_index_sampled_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_index_sampled(&space, 10);
    }

    #[test]
    fn from_index_sampled_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_index_sampled(&space, 30);
    }

    #[test]
    fn from_index_invalid_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn from_index_invalid_index() {
        let space = OptionSpace::new(IndexSpace::new(5));
        testing::check_from_index_invalid(&space);
    }
}

#[cfg(test)]
mod base_feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn num_features_singleton() {
        let space = OptionSpace::new(SingletonSpace::new());
        assert_eq!(space.num_features(), 1);
    }

    #[test]
    fn num_features_index() {
        let space = OptionSpace::new(IndexSpace::new(3));
        assert_eq!(space.num_features(), 4);
    }
}

#[cfg(test)]
mod feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;
    use ndarray::arr1;

    macro_rules! features_tests {
        ($label:ident, $inner:expr, $elem:expr, $expected:expr) => {
            mod $label {
                use super::*;

                #[test]
                fn tensor_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Tensor = space.features(&$elem);
                    assert_eq!(actual, Tensor::of_slice(&$expected));
                }

                #[test]
                fn tensor_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected = Tensor::of_slice(&$expected);
                    let mut out = expected.empty_like();
                    space.features_out(&$elem, &mut out, false);
                    assert_eq!(out, expected);
                }

                #[test]
                fn array_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Array<f32, _> = space.features(&$elem);
                    let expected: Array<f32, _> = arr1(&$expected);
                    assert_eq!(actual, expected);
                }

                #[test]
                fn array_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected: Array<f32, _> = arr1(&$expected);
                    let mut out = Array::from_elem(expected.raw_dim(), f32::NAN);
                    space.features_out(&$elem, &mut out, false);
                    assert_eq!(out, expected);
                }
            }
        };
    }

    features_tests!(singleton_none, SingletonSpace::new(), None, [1.0_f32]);
    features_tests!(singleton_some, SingletonSpace::new(), Some(()), [0.0_f32]);
    features_tests!(
        index_none,
        IndexSpace::new(3),
        None,
        [1.0, 0.0, 0.0, 0.0_f32]
    );
    features_tests!(
        index_some,
        IndexSpace::new(3),
        Some(1),
        [0.0, 0.0, 1.0, 0.0_f32]
    );
}

#[cfg(test)]
mod batch_feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;
    use ndarray::arr2;
    use std::array::IntoIter;

    fn tensor_from_arrays<T: tch::kind::Element, const N: usize, const M: usize>(
        data: [[T; M]; N],
    ) -> Tensor {
        let flat_data: Vec<T> = IntoIter::new(data).map(IntoIter::new).flatten().collect();
        Tensor::of_slice(&flat_data).reshape(&[N as i64, M as i64])
    }

    macro_rules! batch_features_tests {
        ($label:ident, $inner:expr, $elems:expr, $expected:expr) => {
            mod $label {
                use super::*;

                #[test]
                fn tensor_batch_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Tensor = space.batch_features(&$elems);
                    assert_eq!(actual, tensor_from_arrays($expected));
                }

                #[test]
                fn tensor_batch_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected = tensor_from_arrays($expected);
                    let mut out = expected.empty_like();
                    space.batch_features_out(&$elems, &mut out, false);
                    assert_eq!(out, expected);
                }

                #[test]
                fn array_batch_features() {
                    let space = OptionSpace::new($inner);
                    let actual: Array<f32, _> = space.batch_features(&$elems);
                    let expected: Array<f32, _> = arr2(&$expected);
                    assert_eq!(actual, expected);
                }

                #[test]
                fn array_batch_features_out() {
                    let space = OptionSpace::new($inner);
                    let expected: Array<f32, _> = arr2(&$expected);
                    let mut out = Array::from_elem(expected.raw_dim(), f32::NAN);
                    space.batch_features_out(&$elems, &mut out, false);
                    assert_eq!(out, expected);
                }
            }
        };
    }

    batch_features_tests!(
        singleton,
        SingletonSpace::new(),
        [Some(()), None, Some(())],
        [[0.0], [1.0], [0.0_f32]]
    );
    batch_features_tests!(
        index,
        IndexSpace::new(3),
        [Some(1), None, Some(0), Some(2), None],
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0_f32]
        ]
    );
}
