//! Optional space definition.
use super::{ElementRefInto, FeatureSpace, FiniteSpace, Space};
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

/// Feature vectors are:
/// * `1, 0, ..., 0` for `None`
/// * `0, feature_vector(x)` for `Some(x)`.
impl<S: FeatureSpace<Tensor>> PhantomFeatureSpace<Tensor> for OptionSpace<S> {
    fn phantom_num_features(&self) -> usize {
        1 + self.inner.num_features()
    }

    fn phantom_features(&self, element: &Self::Element) -> Tensor {
        let mut out = Tensor::empty(
            &[self.phantom_num_features() as i64],
            (Kind::Float, Device::Cpu),
        );
        self.phantom_features_out(element, &mut out);
        out
    }

    fn phantom_features_out(&self, element: &Self::Element, out: &mut Tensor) {
        let rest_size = self.inner.num_features();
        let [mut first, mut rest]: [Tensor; 2] = out
            .split_with_sizes(&[1, rest_size as i64], -1)
            .try_into()
            .unwrap();
        if let Some(inner_elem) = element {
            let _ = first.fill_(0.0);
            self.inner.features_out(inner_elem, &mut rest);
        } else {
            let _ = first.fill_(1.0);
            let _ = rest.fill_(0.0);
        }
    }

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
            &[elements.len() as i64, self.phantom_num_features() as i64],
            (Kind::Float, Device::Cpu),
        );
        self.phantom_batch_features_out(elements, &mut out, marker);
        out
    }

    fn phantom_batch_features_out<'a, I>(
        &self,
        elements: I,
        out: &mut Tensor,
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

        let _ = out.zero_();
        let _ = first.index_fill_(-1, &Tensor::of_slice(&none_indices), 1.0);
        self.inner
            .batch_features_out(some_elements, &mut rest.i(&Tensor::of_slice(&some_indices)));
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
/// Then in the `FeatureSpace` implementation we use `PhantomFeatureSpace::Element`,
/// which hides the fact that `Element` is an `Option` becasue the compiler accepts
/// lifetime bounds for an arbitrary `S::Element` but not for `Option<S::Element>`...
///
/// # References
/// * <https://users.rust-lang.org/t/lifetime/59967>
/// * <https://github.com/rust-lang/rust/issues/85451>
pub trait PhantomFeatureSpace<T, T2 = T>: Space {
    fn phantom_num_features(&self) -> usize;
    fn phantom_features(&self, element: &Self::Element) -> T;
    fn phantom_features_out(&self, element: &Self::Element, out: &mut T);
    fn phantom_batch_features<'a, I>(
        &self,
        elements: I,
        _marker: PhantomData<&'a Self::Element>,
    ) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
        Self::Element: 'a;
    fn phantom_batch_features_out<'a, I>(
        &self,
        elements: I,
        out: &mut T2,
        _marker: PhantomData<&'a Self::Element>,
    ) where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a;
}

impl<S, T, T2> FeatureSpace<T, T2> for OptionSpace<S>
where
    OptionSpace<S>: PhantomFeatureSpace<T, T2>,
{
    fn num_features(&self) -> usize {
        self.phantom_num_features()
    }
    fn features(&self, element: &Self::Element) -> T {
        self.phantom_features(element)
    }
    fn features_out(&self, element: &Self::Element, out: &mut T) {
        self.phantom_features_out(element, out)
    }
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
    {
        self.phantom_batch_features(elements, PhantomData)
    }
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut T2)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        self.phantom_batch_features_out(elements, out, PhantomData)
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
