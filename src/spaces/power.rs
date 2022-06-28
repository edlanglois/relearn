//! Cartesian power space.
use super::{FeatureSpace, FiniteSpace, LogElementSpace, NonEmptySpace, Space, SubsetOrd};
use crate::logging::{LogError, StatsLogger};
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A Cartesian power of a space: a Cartesian product of `N` copies of the same space.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PowerSpace<S, const N: usize> {
    pub inner: S,
}

impl<S, const N: usize> PowerSpace<S, N> {
    #[inline]
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: Space, const N: usize> Space for PowerSpace<S, N> {
    type Element = [S::Element; N];

    #[inline]
    fn contains(&self, value: &Self::Element) -> bool {
        value.iter().all(|v| self.inner.contains(v))
    }
}

impl<S: SubsetOrd, const N: usize> SubsetOrd for PowerSpace<S, N> {
    #[inline]
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.subset_cmp(&other.inner)
    }
}

impl<S: FiniteSpace, const N: usize> FiniteSpace for PowerSpace<S, N> {
    #[inline]
    fn size(&self) -> usize {
        self.inner
            .size()
            .checked_pow(N.try_into().expect("Size of space is larger than usize"))
            .expect("Size of space is larger than usize")
    }

    #[inline]
    fn to_index(&self, element: &Self::Element) -> usize {
        // The index is obtained by treating the element as a little-endian number
        // when written as a sequence of inner-space indices.
        let inner_size = self.inner.size();
        let mut index = 0;
        for inner_elem in element.iter().rev() {
            index *= inner_size;
            index += self.inner.to_index(inner_elem)
        }
        index
    }

    #[inline]
    fn from_index(&self, mut index: usize) -> Option<Self::Element> {
        let inner_size = self.inner.size();
        let result_elems = array_init::try_array_init(|_| {
            let result_elem = self.inner.from_index(index % inner_size).ok_or(());
            index /= inner_size;
            result_elem
        })
        .ok();
        if index == 0 {
            result_elems
        } else {
            None
        }
    }
}

impl<S, const N: usize> NonEmptySpace for PowerSpace<S, N>
where
    S: NonEmptySpace + Distribution<S::Element>,
{
    #[inline]
    fn some_element(&self) -> Self::Element {
        array_init::array_init(|_| self.inner.some_element())
    }
}

impl<S, const N: usize> Distribution<<Self as Space>::Element> for PowerSpace<S, N>
where
    S: Space + Distribution<S::Element>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        array_init::array_init(|_| self.inner.sample(rng))
    }
}

/// Features are the concatenation of inner feature vectors
impl<S: FeatureSpace, const N: usize> FeatureSpace for PowerSpace<S, N> {
    #[inline]
    fn num_features(&self) -> usize {
        self.inner.num_features() * N
    }

    #[inline]
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        zeroed: bool,
    ) -> &'a mut [F] {
        element.iter().fold(out, |out, inner_elem| {
            self.inner.features_out(inner_elem, out, zeroed)
        })
    }
}

impl<S: Space, const N: usize> LogElementSpace for PowerSpace<S, N> {
    #[inline]
    fn log_element<L: StatsLogger + ?Sized>(
        &self,
        _name: &'static str,
        _element: &Self::Element,
        _logger: &mut L,
    ) -> Result<(), LogError> {
        // Too complex to log
        Ok(())
    }
}

#[cfg(test)]
mod space {
    use super::super::{testing, BooleanSpace, IntervalSpace};
    use super::*;

    #[test]
    fn d0_boolean_contains_empty() {
        let space = PowerSpace::<_, 0>::new(BooleanSpace);
        assert!(space.contains(&[]));
    }

    #[test]
    fn d1_boolean_contains_true() {
        let space = PowerSpace::<_, 1>::new(BooleanSpace);
        assert!(space.contains(&[true]));
    }

    #[test]
    fn d2_boolean_contains_true_false() {
        let space = PowerSpace::<_, 2>::new(BooleanSpace);
        assert!(space.contains(&[true, false]));
    }

    #[test]
    fn d2_interval_contains_point() {
        let space = PowerSpace::<_, 2>::new(IntervalSpace::default());
        assert!(space.contains(&[3.0, -100.0]));
    }

    #[test]
    fn d2_unit_box_not_contains_point() {
        let space = PowerSpace::<_, 2>::new(IntervalSpace::new(0.0, 1.0));
        assert!(!space.contains(&[0.5, 2.1]));
    }

    #[test]
    fn d0_boolean_contains_samples() {
        let space = PowerSpace::<_, 0>::new(BooleanSpace);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn d1_boolean_contains_samples() {
        let space = PowerSpace::<_, 1>::new(BooleanSpace);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn d2_boolean_contains_samples() {
        let space = PowerSpace::<_, 2>::new(BooleanSpace);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn d2_interval_contains_samples() {
        let space = PowerSpace::<_, 2>::new(IntervalSpace::<f64>::default());
        testing::check_contains_samples(&space, 10);
    }
}

#[cfg(test)]
mod subset_cmp {
    use super::super::IntervalSpace;
    use super::*;

    #[test]
    fn d0_interval_eq() {
        assert_eq!(
            PowerSpace::<_, 0>::new(IntervalSpace::<f64>::default()),
            PowerSpace::<_, 0>::new(IntervalSpace::<f64>::default())
        );
    }

    #[test]
    fn d2_interval_eq() {
        assert_eq!(
            PowerSpace::<_, 2>::new(IntervalSpace::<f64>::default()),
            PowerSpace::<_, 2>::new(IntervalSpace::<f64>::default())
        );
    }

    #[test]
    fn d2_interval_ne() {
        assert_ne!(
            PowerSpace::<_, 2>::new(IntervalSpace::default()),
            PowerSpace::<_, 2>::new(IntervalSpace::new(0.0, 1.0))
        );
    }

    #[test]
    fn d2_interval_cmp_equal() {
        assert_eq!(
            PowerSpace::<_, 2>::new(IntervalSpace::<f64>::default())
                .subset_cmp(&PowerSpace::new(IntervalSpace::default())),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn d2_interval_strict_subset() {
        assert!(PowerSpace::<_, 2>::new(IntervalSpace::new(0.0, 1.0))
            .strict_subset_of(&PowerSpace::<_, 2>::new(IntervalSpace::default())));
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::{testing, BooleanSpace};
    use super::*;

    #[test]
    fn d3_boolean_from_to_index_iter_size() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn d3_boolean_from_to_index_random() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        testing::check_from_to_index_random(&space, 10);
    }

    #[test]
    fn d3_boolean_from_index_sampled() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        testing::check_from_index_sampled(&space, 10);
    }

    #[test]
    fn d3_boolean_from_index_invalid() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn d0_boolean_size() {
        let space = PowerSpace::<_, 0>::new(BooleanSpace);
        assert_eq!(space.size(), 1);
    }

    #[test]
    fn d0_boolean_to_index() {
        let space = PowerSpace::<_, 0>::new(BooleanSpace);
        assert_eq!(space.to_index(&[]), 0);
    }

    #[test]
    fn d0_boolean_from_index() {
        let space = PowerSpace::<_, 0>::new(BooleanSpace);
        assert_eq!(space.from_index(0), Some([]));
    }

    #[test]
    fn d3_boolean_size() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        assert_eq!(space.size(), 8);
    }

    #[test]
    fn d3_boolean_to_index() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        assert_eq!(space.to_index(&[false, false, false]), 0);
        assert_eq!(space.to_index(&[true, false, false]), 1);
        assert_eq!(space.to_index(&[false, true, false]), 2);
        assert_eq!(space.to_index(&[false, false, true]), 4);
        assert_eq!(space.to_index(&[true, true, true]), 7);
    }

    #[test]
    fn d3_boolean_from_index() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        assert_eq!(space.from_index(0), Some([false, false, false]));
        assert_eq!(space.from_index(4), Some([false, false, true]));
        assert_eq!(space.from_index(7), Some([true, true, true]));
    }
}

#[cfg(test)]
mod feature_space {
    use super::super::{BooleanSpace, IndexSpace};
    use super::*;

    #[test]
    fn d3_boolean_num_features() {
        let space = PowerSpace::<_, 3>::new(BooleanSpace);
        assert_eq!(space.num_features(), 3);
    }

    #[test]
    fn d3_index2_num_features() {
        let space = PowerSpace::<_, 3>::new(IndexSpace::new(2));
        assert_eq!(space.num_features(), 6);
    }

    features_tests!(d0_boolean, PowerSpace::<_, 0>::new(BooleanSpace), [], []);
    features_tests!(
        d2_boolean,
        PowerSpace::<_, 2>::new(BooleanSpace),
        [true, false],
        [1.0, 0.0]
    );
    features_tests!(
        d2_index2,
        PowerSpace::<_, 2>::new(IndexSpace::new(2)),
        [1, 0],
        [0.0, 1.0, 1.0, 0.0]
    );

    batch_features_tests!(
        batch_d0_index2,
        PowerSpace::<_, 0>::new(IndexSpace::new(2)),
        [[], []],
        [[], []]
    );
    batch_features_tests!(
        batch_d2_index2,
        PowerSpace::<_, 2>::new(IndexSpace::new(2)),
        [[0, 0], [1, 1], [1, 0]],
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    );
}
