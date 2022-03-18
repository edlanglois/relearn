//! Option space definition.
use super::{ElementRefInto, FeatureSpace, FiniteSpace, NonEmptySpace, Space, SubsetOrd};
use crate::logging::Loggable;
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

/// A space whose elements are either `None` or `Some(inner_elem)`.
///
/// The feature vectors are
/// * `[1, 0, ..., 0]` for `None`
/// * `[0, inner_feature_vector(x)]` for `Some(x)`.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct OptionSpace<S> {
    pub inner: S,
}

impl<S> OptionSpace<S> {
    #[inline]
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: fmt::Display> fmt::Display for OptionSpace<S> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OptionSpace<{}>", self.inner)
    }
}

impl<S: Space> Space for OptionSpace<S> {
    type Element = Option<S::Element>;

    #[inline]
    fn contains(&self, value: &Self::Element) -> bool {
        match value {
            None => true,
            Some(inner_value) => self.inner.contains(inner_value),
        }
    }
}

impl<S: SubsetOrd> SubsetOrd for OptionSpace<S> {
    #[inline]
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.subset_cmp(&other.inner)
    }
}

impl<S: FiniteSpace> FiniteSpace for OptionSpace<S> {
    #[inline]
    fn size(&self) -> usize {
        1 + self.inner.size()
    }

    #[inline]
    fn to_index(&self, element: &Self::Element) -> usize {
        match element {
            None => 0,
            Some(inner_elem) => 1 + self.inner.to_index(inner_elem),
        }
    }

    #[inline]
    fn from_index(&self, index: usize) -> Option<Self::Element> {
        if index == 0 {
            Some(None)
        } else {
            Some(Some(self.inner.from_index(index - 1)?))
        }
    }
}

impl<S: Space> NonEmptySpace for OptionSpace<S> {
    #[inline]
    fn some_element(&self) -> Self::Element {
        None
    }
}

/// Features are `[0.0, features(x)..]` for `Some(x)` and `[1.0, 0.0, ..., 0.0]` for `None`.
impl<S: FeatureSpace> FeatureSpace for OptionSpace<S> {
    #[inline]
    fn num_features(&self) -> usize {
        1 + self.inner.num_features()
    }

    #[inline]
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        zeroed: bool,
    ) -> &'a mut [F] {
        match element {
            None => {
                let end = self.inner.num_features() + 1;
                out[0] = F::one();
                if !zeroed {
                    out[1..end].fill(F::zero());
                }
                &mut out[end..]
            }
            Some(inner_elem) => {
                out[0] = F::zero();
                self.inner.features_out(inner_elem, &mut out[1..], zeroed)
            }
        }
    }
}

impl<S> Distribution<<Self as Space>::Element> for OptionSpace<S>
where
    S: Space + Distribution<S::Element>,
{
    #[inline]
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
    #[inline]
    fn elem_ref_into(&self, _element: &Self::Element) -> Loggable {
        // No clear way to convert structured elements into Loggable
        Loggable::Nothing
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
mod feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;

    mod singleton {
        use super::*;

        #[test]
        fn num_features() {
            let space = OptionSpace::new(SingletonSpace::new());
            assert_eq!(space.num_features(), 1);
        }

        features_tests!(none, OptionSpace::new(SingletonSpace::new()), None, [1.0]);
        features_tests!(
            some,
            OptionSpace::new(SingletonSpace::new()),
            Some(()),
            [0.0]
        );
        batch_features_tests!(
            batch,
            OptionSpace::new(SingletonSpace::new()),
            [Some(()), None, Some(())],
            [[0.0], [1.0], [0.0]]
        );
    }

    mod index {
        use super::*;

        #[test]
        fn num_features() {
            let space = OptionSpace::new(IndexSpace::new(3));
            assert_eq!(space.num_features(), 4);
        }

        features_tests!(
            none,
            OptionSpace::new(IndexSpace::new(3)),
            None,
            [1.0, 0.0, 0.0, 0.0]
        );
        features_tests!(
            some,
            OptionSpace::new(IndexSpace::new(3)),
            Some(1),
            [0.0, 0.0, 1.0, 0.0]
        );
        batch_features_tests!(
            batch,
            OptionSpace::new(IndexSpace::new(3)),
            [Some(1), None, Some(0), Some(2), None],
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0]
            ]
        );
    }
}
