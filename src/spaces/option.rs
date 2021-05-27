//! Optional space definition.
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, ElementRefInto, FeatureSpace,
    FeatureSpaceOut, FiniteSpace, Space,
};
use crate::logging::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
use std::convert::TryInto;
use std::fmt;
use std::marker::PhantomData;
use tch::{Device, IndexOp, Kind, Tensor};

/// A space whose elements are either `None` or `Some(inner_elem)`.
#[derive(Debug, Clone)]
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

/// Feature vectors are:
/// * `1, 0, ..., 0` for `None`
/// * `0, feature_vector(x)` for `Some(x)`.
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

impl<S: BatchFeatureSpaceOut<Tensor>> PhantomBatchFeatureSpace<Tensor> for OptionSpace<S> {
    fn phantom_batch_features<'a, I>(
        &self,
        elements: I,
        marker: PhantomData<&'a Self::Element>,
    ) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
    {
        let elements = elements.into_iter();
        let mut out = Tensor::empty(
            &[elements.len() as i64, self.num_features() as i64],
            (Kind::Float, Device::Cpu),
        );
        self.phantom_batch_features_out(elements, &mut out, false, marker);
        out
    }
}

impl<S: BatchFeatureSpaceOut<Tensor>> PhantomBatchFeatureSpaceOut<Tensor> for OptionSpace<S> {
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
        let [mut first, rest]: [Tensor; 2] = out
            .split_with_sizes(&[1, rest_size as i64], -1)
            .try_into()
            .unwrap();

        if !zeroed {
            let _ = out.zero_();
        }
        let _ = first.index_fill_(-2, &Tensor::of_slice(&none_indices), 1.0);
        // FIXME: This creates a copy not a view
        let mut some_rest = rest.i(&Tensor::of_slice(&some_indices));
        self.inner
            .batch_features_out(some_elements, &mut some_rest, true);
    }
}

impl<S> Distribution<<Self as Space>::Element> for OptionSpace<S>
where
    S: Space + Distribution<<S as Space>::Element>,
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
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
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
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
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
mod feature_space_tensor {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;
    use std::cmp::PartialEq;
    use std::fmt::Debug;

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

    fn check_option_features<S, T>(inner: S, element: &Option<S::Element>, expected: &T)
    where
        S: FeatureSpace<T>,
        // These ought to be implied by S: FeatureSpace<T> but the whole PhantomFeatureSpace
        // hack hides this inference from the compiler.
        // I think the `where self: PhantomFeatureSpace<T, T2>` is the problem.
        OptionSpace<S>: FeatureSpace<T> + Space<Element = Option<S::Element>>,
        T: Debug + PartialEq,
    {
        let space = OptionSpace::new(inner);
        assert_eq!(&space.features(element), expected);
    }

    fn check_option_features_out<S>(inner: S, element: &Option<S::Element>, expected: &Tensor)
    where
        S: FeatureSpaceOut<Tensor>,
        // These ought to be implied by S: FeatureSpaceOut<T> but the whole PhantomFeatureSpace
        // hack hides this inference from the compiler.
        // I think the `where self: PhantomFeatureSpace<T, T2>` is the problem.
        OptionSpace<S>: FeatureSpaceOut<Tensor> + Space<Element = Option<S::Element>>,
    {
        let space = OptionSpace::new(inner);
        let mut out = expected.empty_like();
        space.features_out(element, &mut out, false);
        assert_eq!(&out, expected);
    }

    #[test]
    fn features_singleton_none() {
        check_option_features(SingletonSpace::new(), &None, &Tensor::of_slice(&[1.0_f32]));
    }

    #[test]
    fn features_out_singleton_none() {
        check_option_features_out(SingletonSpace::new(), &None, &Tensor::of_slice(&[1.0_f32]));
    }

    #[test]
    fn features_singleton_some() {
        check_option_features(
            SingletonSpace::new(),
            &Some(()),
            &Tensor::of_slice(&[0.0_f32]),
        );
    }

    #[test]
    fn features_out_singleton_some() {
        check_option_features_out(
            SingletonSpace::new(),
            &Some(()),
            &Tensor::of_slice(&[0.0_f32]),
        );
    }

    #[test]
    fn features_index_none() {
        check_option_features(
            IndexSpace::new(3),
            &None,
            &Tensor::of_slice(&[1.0_f32, 0.0, 0.0, 0.0]),
        )
    }

    #[test]
    fn features_out_index_none() {
        check_option_features_out(
            IndexSpace::new(3),
            &None,
            &Tensor::of_slice(&[1.0_f32, 0.0, 0.0, 0.0]),
        )
    }

    #[test]
    fn features_index_some() {
        check_option_features(
            IndexSpace::new(3),
            &Some(1),
            &Tensor::of_slice(&[0.0_f32, 0.0, 1.0, 0.0]),
        )
    }

    #[test]
    fn features_out_index_some() {
        check_option_features_out(
            IndexSpace::new(3),
            &Some(1),
            &Tensor::of_slice(&[0.0_f32, 0.0, 1.0, 0.0]),
        )
    }

    fn check_option_batch_features<S, T>(inner: S, elements: &[Option<S::Element>], expected: &T)
    where
        S: FeatureSpace<T>,
        // These ought to be implied by S: FeatureSpace<T> but the whole PhantomFeatureSpace
        // hack hides this inference from the compiler.
        // I think the `where self: PhantomFeatureSpace<T, T2>` is the problem.
        OptionSpace<S>: BatchFeatureSpace<T> + Space<Element = Option<S::Element>>,
        T: Debug + PartialEq,
    {
        let space = OptionSpace::new(inner);
        assert_eq!(&space.batch_features(elements), expected);
    }

    fn check_option_batch_features_out<S>(
        inner: S,
        elements: &[Option<S::Element>],
        expected: &Tensor,
    ) where
        S: FeatureSpaceOut<Tensor>,
        // These ought to be implied by S: FeatureSpaceOut<T> but the whole PhantomFeatureSpace
        // hack hides this inference from the compiler.
        // I think the `where self: PhantomFeatureSpace<T, T2>` is the problem.
        OptionSpace<S>: BatchFeatureSpaceOut<Tensor> + Space<Element = Option<S::Element>>,
    {
        let space = OptionSpace::new(inner);
        let mut out = expected.empty_like();
        space.batch_features_out(elements, &mut out, false);
        assert_eq!(&out, expected);
    }

    #[test]
    fn batch_features_singleton() {
        check_option_batch_features(
            SingletonSpace::new(),
            &[Some(()), None, Some(())],
            &Tensor::of_slice(&[0.0_f32, 1.0, 0.0]).view((3, 1)),
        );
    }

    #[test]
    fn batch_features_out_singleton() {
        check_option_batch_features_out(
            SingletonSpace::new(),
            &[Some(()), None, Some(())],
            &Tensor::of_slice(&[0.0_f32, 1.0, 0.0]).view((3, 1)),
        );
    }

    #[test]
    fn batch_features_index() {
        // Note: Currently fails because the complicated indexing used when generating the input
        // for inner.batch_features_out results in a copy rather than a view.
        check_option_batch_features(
            IndexSpace::new(3),
            &[Some(1), None, Some(0), Some(2), None],
            &Tensor::of_slice(&[
                0.0, 0.0, 1.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 0.0_f32,
            ])
            .view((5, 4)),
        );
    }

    #[test]
    fn batch_features_out_index() {
        // Note: Currently fails because the complicated indexing used when generating the input
        // for inner.batch_features_out results in a copy rather than a view.
        check_option_batch_features_out(
            IndexSpace::new(3),
            &[Some(1), None, Some(0), Some(2), None],
            &Tensor::of_slice(&[
                0.0, 0.0, 1.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 0.0_f32,
            ])
            .view((5, 4)),
        );
    }
}
