//! Wrap spaces to have non-empty feature vectors.
use super::{FeatureSpace, FiniteSpace, LogElementSpace, NonEmptySpace, Space, SubsetOrd};
use crate::logging::{LogError, StatsLogger};
use crate::utils::num_array::{BuildFromArray1D, BuildFromArray2D, NumArray1D, NumArray2D};
use ndarray::{s, ArrayBase, DataMut, Ix2};
use num_traits::{Float, Zero};
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

// Note: This could be renamed to something like TorchSpace if other helper changes end up being
// requried to satisfy torch methods.

/// Wrapper space with a feature vector length of at least 1.
///
/// Any generated features have value `0`.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct NonEmptyFeatures<S> {
    inner: S,
}

impl<S> NonEmptyFeatures<S> {
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: fmt::Display> fmt::Display for NonEmptyFeatures<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NonEmptyFeatures<{}>", self.inner)
    }
}

impl<S: Space> Space for NonEmptyFeatures<S> {
    type Element = S::Element;

    fn contains(&self, value: &Self::Element) -> bool {
        self.inner.contains(value)
    }
}

impl<S: SubsetOrd> SubsetOrd for NonEmptyFeatures<S> {
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.subset_cmp(&other.inner)
    }
}

impl<S: FiniteSpace> FiniteSpace for NonEmptyFeatures<S> {
    fn size(&self) -> usize {
        self.inner.size()
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        self.inner.to_index(element)
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        self.inner.from_index(index)
    }
}

impl<S: NonEmptySpace> NonEmptySpace for NonEmptyFeatures<S> {
    fn some_element(&self) -> Self::Element {
        self.inner.some_element()
    }
}

impl<S: Space + Distribution<S::Element>> Distribution<<Self as Space>::Element>
    for NonEmptyFeatures<S>
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        self.inner.sample(rng)
    }
}

/// Features are either the inner features if non-empty or `[0.0]`.
impl<S: FeatureSpace> FeatureSpace for NonEmptyFeatures<S> {
    #[inline]
    fn num_features(&self) -> usize {
        self.inner.num_features().max(1)
    }

    #[inline]
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        zeroed: bool,
    ) -> &'a mut [F] {
        let out_len = out.len();
        let rest = self.inner.features_out(element, out, zeroed);
        if rest.len() == out_len {
            // No features have been written; write a 0
            rest[0] = Zero::zero();
            &mut rest[1..]
        } else {
            // features written by inner
            rest
        }
    }

    #[inline]
    fn features<T>(&self, element: &Self::Element) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        let num_inner_features = self.inner.num_features();
        let mut array = T::Array::zeros(num_inner_features.max(1));
        if num_inner_features > 0 {
            self.features_out(element, array.as_slice_mut(), true);
        }
        array.into()
    }

    #[inline]
    fn batch_features_out<'a, I, A>(&self, elements: I, out: &mut ArrayBase<A, Ix2>, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
        let num_inner_features = self.inner.num_features();
        if num_inner_features > 0 {
            self.inner.batch_features_out(elements, out, zeroed)
        } else if !zeroed {
            out.slice_mut(s![.., 0]).fill(Zero::zero())
        }
    }

    #[inline]
    fn batch_features<'a, I, T>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
        T: BuildFromArray2D,
        <T::Array as NumArray2D>::Elem: Float,
    {
        let num_inner_features = self.inner.num_features();
        if num_inner_features > 0 {
            self.inner.batch_features(elements)
        } else {
            T::Array::zeros((elements.into_iter().len(), 1)).into()
        }
    }
}

impl<S: LogElementSpace> LogElementSpace for NonEmptyFeatures<S> {
    #[inline]
    fn log_element<L: StatsLogger + ?Sized>(
        &self,
        name: &'static str,
        element: &Self::Element,
        logger: &mut L,
    ) -> Result<(), LogError> {
        self.inner.log_element(name, element, logger)
    }
}

#[cfg(test)]
mod space {
    use super::super::{testing, IndexSpace, SingletonSpace};
    use super::*;

    #[test]
    fn contains_samples_singleton() {
        let space = NonEmptyFeatures::new(SingletonSpace::new());
        testing::check_contains_samples(&space, 100);
    }

    #[test]
    fn contains_samples_index() {
        let space = NonEmptyFeatures::new(IndexSpace::new(5));
        testing::check_contains_samples(&space, 100);
    }
}

#[cfg(test)]
mod base_feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;
    #[test]
    fn num_features_wrap_0() {
        let inner = SingletonSpace::new();
        assert_eq!(inner.num_features(), 0);
        let space = NonEmptyFeatures::new(inner);
        assert_eq!(space.num_features(), 1);
    }

    #[test]
    fn num_features_wrap_2() {
        let inner = IndexSpace::new(2);
        assert_eq!(inner.num_features(), 2);
        let space = NonEmptyFeatures::new(inner);
        assert_eq!(space.num_features(), 2);
    }
}

#[cfg(test)]
mod feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;

    mod wrap_singleton {
        use super::*;

        const fn space() -> NonEmptyFeatures<SingletonSpace> {
            NonEmptyFeatures::new(SingletonSpace::new())
        }

        #[test]
        fn num_features() {
            let space = space();
            assert_eq!(space.num_features(), 1);
        }

        features_tests!(f, space(), (), [0.0]);
        batch_features_tests!(b, space(), [(), (), ()], [[0.0], [0.0], [0.0]]);
    }

    mod wrap_index_1 {
        use super::*;

        const fn space() -> NonEmptyFeatures<IndexSpace> {
            NonEmptyFeatures::new(IndexSpace::new(1))
        }

        #[test]
        fn num_features() {
            let space = space();
            assert_eq!(space.num_features(), 1);
        }

        features_tests!(f, space(), 0, [1.0]);
        batch_features_tests!(b, space(), [0, 0, 0], [[1.0], [1.0], [1.0]]);
    }

    mod wrap_index_2 {
        use super::*;

        const fn space() -> NonEmptyFeatures<IndexSpace> {
            NonEmptyFeatures::new(IndexSpace::new(2))
        }

        #[test]
        fn num_features() {
            let space = space();
            assert_eq!(space.num_features(), 2);
        }

        features_tests!(f0, space(), 0, [1.0, 0.0]);
        features_tests!(f1, space(), 1, [0.0, 1.0]);
        batch_features_tests!(b, space(), [1, 0, 1], [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]);
    }
}
