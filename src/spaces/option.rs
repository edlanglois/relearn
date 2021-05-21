//! Optional space definition.
use super::{ElementRefInto, FeatureSpace, FiniteSpace, Space};
use crate::logging::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
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
        let out = Tensor::zeros(
            &[self.phantom_num_features() as i64],
            (Kind::Float, Device::Cpu),
        );
        if let Some(inner_elem) = element {
            out.i(1..).copy_(&self.inner.features(inner_elem));
        } else {
            let _ = out.i(0).fill_(1.0);
        }
        out
    }

    fn phantom_batch_features<'a, I>(
        &self,
        elements: I,
        _: PhantomData<&'a Self::Element>,
    ) -> Tensor
    where
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
        let batch_size = none_indices.len() + some_indices.len();
        let out = Tensor::zeros(
            &[batch_size as i64, self.phantom_num_features() as i64],
            (Kind::Float, Device::Cpu),
        );
        let _ = out.i((&Tensor::of_slice(&none_indices), 0)).fill_(1.0);
        out.i((&Tensor::of_slice(&some_indices), 1..))
            .copy_(&self.inner.batch_features(some_elements));
        out
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
    fn phantom_batch_features<'a, I>(
        &self,
        elements: I,
        _marker: PhantomData<&'a Self::Element>,
    ) -> T2
    where
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
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        self.phantom_batch_features(elements, PhantomData)
    }
}

#[cfg(test)]
mod option_space {
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
