//! `IntervalSpace` definition
use super::{ElementRefInto, FeatureSpace, NonEmptySpace, ReprSpace, Space, SubsetOrd};
use crate::logging::LogValue;
use num_traits::{Bounded, Float, ToPrimitive};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::{Gamma, StandardNormal};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::{fmt, slice};
use tch::Tensor;

/// A closed interval on bounded, partially ordered types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntervalSpace<T> {
    pub low: T,
    pub high: T,
}

impl<T: PartialOrd> IntervalSpace<T> {
    #[inline]
    pub fn new(low: T, high: T) -> Self {
        assert!(low <= high, "require low <= high");
        Self { low, high }
    }
}

/// The default interval is the full real number line.
impl<T: Bounded> Default for IntervalSpace<T> {
    #[inline]
    fn default() -> Self {
        Self {
            low: T::min_value(),
            high: T::max_value(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for IntervalSpace<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IntervalSpace({}, {})", self.low, self.high)
    }
}

impl<T: Bounded + PartialOrd + Clone + Send> Space for IntervalSpace<T> {
    type Element = T;

    #[inline]
    fn contains(&self, value: &Self::Element) -> bool {
        &self.low <= value && value <= &self.high
    }
}

impl<T: PartialOrd> SubsetOrd for IntervalSpace<T> {
    #[inline]
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        use Ordering::*;
        match (
            self.low.partial_cmp(&other.low),
            self.high.partial_cmp(&other.high),
        ) {
            (Some(Equal), Some(Equal)) => Some(Equal),
            (Some(Equal | Greater), Some(Equal | Less)) => Some(Less),
            (Some(Equal | Less), Some(Equal | Greater)) => Some(Greater),
            _ => None,
        }
    }
}

impl<T: Bounded + PartialOrd + Clone + Send> NonEmptySpace for IntervalSpace<T> {
    #[inline]
    fn some_element(&self) -> Self::Element {
        self.low.clone()
    }
}

/// Represent elements as the same type in a tensor.
impl<T> ReprSpace<Tensor> for IntervalSpace<T>
where
    T: Bounded + PartialOrd + tch::kind::Element + Clone + Send,
{
    #[inline]
    fn repr(&self, element: &Self::Element) -> Tensor {
        Tensor::of_slice(slice::from_ref(element)).squeeze_dim_(0)
    }

    #[inline]
    fn batch_repr<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        let elements: Vec<_> = elements.into_iter().cloned().collect();
        Tensor::of_slice(&elements)
    }
}

/// Features are `[f]` for element `f`.
impl<T: Bounded + PartialOrd + ToPrimitive + Clone + Send> FeatureSpace for IntervalSpace<T> {
    #[inline]
    fn num_features(&self) -> usize {
        1
    }

    #[inline]
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        _zeroed: bool,
    ) -> &'a mut [F] {
        out[0] = F::from(element.clone()).expect("could not convert element to float");
        &mut out[1..]
    }
}

impl Distribution<f32> for IntervalSpace<f32> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        match (
            self.low > Bounded::min_value(),
            self.high < Bounded::max_value(),
        ) {
            (true, true) => rng.gen_range(self.low..=self.high),
            (true, false) => self.low + Gamma::new(1.0, 1.0).unwrap().sample(rng),
            (false, true) => self.high - Gamma::new(1.0, 1.0).unwrap().sample(rng),
            (false, false) => StandardNormal.sample(rng),
        }
    }
}

impl Distribution<f64> for IntervalSpace<f64> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        match (
            self.low > Bounded::min_value(),
            self.high < Bounded::max_value(),
        ) {
            (true, true) => rng.gen_range(self.low..=self.high),
            (true, false) => self.low + Gamma::new(1.0, 1.0).unwrap().sample(rng),
            (false, true) => self.high - Gamma::new(1.0, 1.0).unwrap().sample(rng),
            (false, false) => StandardNormal.sample(rng),
        }
    }
}

impl<T: Bounded + PartialOrd + Into<f64> + Clone + Send> ElementRefInto<LogValue>
    for IntervalSpace<T>
{
    #[inline]
    fn elem_ref_into(&self, element: &Self::Element) -> LogValue {
        LogValue::Scalar(element.clone().into())
    }
}

#[cfg(test)]
mod space {
    use super::super::testing;
    use super::*;

    #[test]
    fn unit_contains_0() {
        let space = IntervalSpace::new(0.0, 1.0);
        assert!(space.contains(&0.0));
    }

    #[test]
    fn unit_contains_half() {
        let space = IntervalSpace::new(0.0, 1.0);
        assert!(space.contains(&0.5));
    }

    #[test]
    fn unit_contains_1() {
        let space = IntervalSpace::new(0.0, 1.0);
        assert!(space.contains(&1.0));
    }

    #[test]
    fn unit_not_contains_2() {
        let space = IntervalSpace::new(0.0, 1.0);
        assert!(!space.contains(&2.0));
    }

    #[test]
    fn unit_not_contains_neg_1() {
        let space = IntervalSpace::new(0.0, 1.0);
        assert!(!space.contains(&-1.0));
    }

    #[test]
    fn unit_contains_samples() {
        let space = IntervalSpace::new(0.0, 1.0);
        testing::check_contains_samples(&space, 20);
    }

    #[test]
    fn unbounded_contains_0() {
        let space = IntervalSpace::default();
        assert!(space.contains(&0.0));
    }

    #[test]
    fn unbounded_contains_100() {
        let space = IntervalSpace::default();
        assert!(space.contains(&100.0));
    }

    #[test]
    fn unbounded_contains_neg_1() {
        let space = IntervalSpace::default();
        assert!(space.contains(&-1.0));
    }

    #[test]
    fn unbounded_not_contains_inf() {
        let space = IntervalSpace::default();
        assert!(!space.contains(&f64::infinity()));
    }

    #[test]
    fn unbounded_not_contains_nan() {
        let space = IntervalSpace::default();
        assert!(!space.contains(&f64::nan()));
    }

    #[test]
    fn unbounded_contains_samples() {
        let space = IntervalSpace::<f64>::default();
        testing::check_contains_samples(&space, 20);
    }

    #[test]
    fn half_contains_lower_bound() {
        let space = IntervalSpace::new(2.0, f64::infinity());
        assert!(space.contains(&2.0));
    }

    #[test]
    fn half_contains_samples() {
        let space = IntervalSpace::new(2.0, f64::infinity());
        testing::check_contains_samples(&space, 20);
    }

    #[test]
    fn point_contains_point() {
        let space = IntervalSpace::new(2.0, 2.0);
        assert!(space.contains(&2.0));
    }

    #[test]
    fn point_not_contains_outside() {
        let space = IntervalSpace::new(2.0, 2.0);
        assert!(!space.contains(&2.1));
    }

    #[test]
    fn point_contains_samples() {
        let space = IntervalSpace::new(2.0, 2.0);
        testing::check_contains_samples(&space, 5);
    }

    #[test]
    #[should_panic]
    fn empty_interval_panics() {
        let _ = IntervalSpace::new(1.0, 0.0);
    }
}

#[cfg(test)]
mod subset_ord {
    use super::*;

    #[test]
    fn same_eq() {
        assert_eq!(IntervalSpace::new(0.0, 1.0), IntervalSpace::new(0.0, 1.0));
    }

    #[test]
    fn same_cmp_equal() {
        assert_eq!(
            IntervalSpace::new(0.0, 1.0).subset_cmp(&IntervalSpace::new(0.0, 1.0)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn different_ne() {
        assert!(IntervalSpace::new(0.0, 1.0) != IntervalSpace::new(0.5, 1.0));
    }

    #[test]
    fn strict_subset() {
        assert!(IntervalSpace::new(0.0, 1.0).strict_subset_of(&IntervalSpace::new(-1.0, 1.0)));
    }

    #[test]
    fn same_subset() {
        assert!(IntervalSpace::new(0.0, 1.0).subset_of(&IntervalSpace::new(0.0, 1.0)));
    }

    #[test]
    fn strict_superset() {
        assert!(IntervalSpace::new(0.0, 1.0).strict_superset_of(&IntervalSpace::new(0.2, 0.8)));
    }

    #[test]
    fn disjoint_incomparable() {
        assert_eq!(
            IntervalSpace::new(0.0, 1.0).subset_cmp(&IntervalSpace::new(2.0, 3.0)),
            None
        );
    }

    #[test]
    fn intersecting_incomparable() {
        assert_eq!(
            IntervalSpace::new(0.0, 2.0).subset_cmp(&IntervalSpace::new(1.0, 3.0)),
            None
        );
    }
}

#[cfg(test)]
mod feature_space {
    use super::*;

    mod unit {
        use super::*;

        #[test]
        fn num_features() {
            let space = IntervalSpace::new(0.0, 1.0);
            assert_eq!(space.num_features(), 1);
        }

        features_tests!(zero, IntervalSpace::new(0.0, 1.0), 0.0, [0.0]);
        features_tests!(one, IntervalSpace::new(0.0, 1.0), 1.0, [1.0]);
        features_tests!(half, IntervalSpace::new(0.0, 1.0), 0.5, [0.5]);
        batch_features_tests!(
            zero_one_half,
            IntervalSpace::new(0.0, 1.0),
            [0.0, 1.0, 0.5],
            [[0.0], [1.0], [0.5]]
        );
    }

    mod neg_pos_one {
        use super::*;

        #[test]
        fn num_features() {
            let space = IntervalSpace::new(-1.0, 1.0);
            assert_eq!(space.num_features(), 1);
        }

        features_tests!(zero, IntervalSpace::new(-1.0, 1.0), 0.0, [0.0]);
        features_tests!(neg_one, IntervalSpace::new(-1.0, 1.0), -1.0, [-1.0]);
        batch_features_tests!(
            zero_neg_one,
            IntervalSpace::new(-1.0, 1.0),
            [0.0, -1.0],
            [[0.0], [-1.0]]
        );
    }

    mod unbounded {
        use super::*;

        #[test]
        fn num_features() {
            let space = IntervalSpace::<f64>::default();
            assert_eq!(space.num_features(), 1);
        }

        features_tests!(zero, IntervalSpace::default(), 0.0, [0.0]);
        features_tests!(ten, IntervalSpace::default(), 10.0, [10.0]);
        batch_features_tests!(
            zero_ten,
            IntervalSpace::default(),
            [0.0, 10.0],
            [[0.0], [10.0]]
        );
    }
}
