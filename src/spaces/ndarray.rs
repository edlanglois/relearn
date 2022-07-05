use super::{FeatureSpace, LogElementSpace, NonEmptySpace, Space, SubsetOrd};
use crate::logging::{LogError, StatsLogger};
use ndarray::{Array, Dimension, IntoDimension, Ix1, Ix2, Ix3};
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A space of n-dimensional [`Array`s](Array) of elements from a single space.
///
/// Similar to [`PowerSpace`](super::PowerSpace) but with heap-allocated allocated,
/// multidimensional elements.
///
/// # Example
/// ```
/// use relearn::spaces::NdArraySpace;
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NdArraySpace<S, D> {
    /// Inner element space
    pub inner: S,
    /// Dimensions of the array
    pub dim: D,
}

/// One-dimensional [`NdArraySpace`]
pub type Array1Space<S> = NdArraySpace<S, Ix1>;
/// Two-dimensional [`NdArraySpace`]
pub type Array2Space<S> = NdArraySpace<S, Ix2>;
/// Three-dimensional [`NdArraySpace`]
pub type Array3Space<S> = NdArraySpace<S, Ix3>;

impl<S, D> NdArraySpace<S, D> {
    #[inline]
    pub fn new<E: IntoDimension<Dim = D>>(inner: S, shape: E) -> Self {
        Self {
            inner,
            dim: shape.into_dimension(),
        }
    }
}

impl<S: Space, D: Dimension> Space for NdArraySpace<S, D> {
    type Element = Array<S::Element, D>;

    #[inline]
    fn contains(&self, value: &Self::Element) -> bool {
        println!(
            "(value.raw_dim() == self.dim) {:?}",
            (value.raw_dim() == self.dim)
        );
        println!(
            "value.iter().all(|e| self.inner.contains(e)) {:?}",
            value.iter().all(|e| self.inner.contains(e))
        );

        (value.raw_dim() == self.dim) && value.iter().all(|e| self.inner.contains(e))
    }
}

impl<S: SubsetOrd, D: Dimension> SubsetOrd for NdArraySpace<S, D> {
    #[inline]
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.dim == other.dim {
            self.inner.subset_cmp(&other.inner)
        } else {
            None
        }
    }
}

impl<S: NonEmptySpace, D: Dimension> NonEmptySpace for NdArraySpace<S, D> {
    #[inline]
    fn some_element(&self) -> Self::Element {
        Array::from_elem(self.dim.clone(), self.inner.some_element())
    }
}

impl<S, D: Dimension> Distribution<<Self as Space>::Element> for NdArraySpace<S, D>
where
    S: Space + Distribution<S::Element>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        Array::from_shape_simple_fn(self.dim.clone(), || self.inner.sample(rng))
    }
}

/// Features are the concatenation of inner feature vectors
impl<S: FeatureSpace, D: Dimension> FeatureSpace for NdArraySpace<S, D> {
    #[inline]
    fn num_features(&self) -> usize {
        self.inner.num_features() * self.dim.size()
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

impl<S: LogElementSpace, D: Dimension> LogElementSpace for NdArraySpace<S, D> {
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
    use super::super::{testing, IntervalSpace};
    use super::*;

    #[test]
    fn d0_singleton_interval_contains_inner_elem() {
        let space = NdArraySpace::new(IntervalSpace::new(0.5, 1.5), ());
        assert!(space.contains(&Array::ones(())))
    }

    #[test]
    fn d0_singleton_interval_not_contains_inner_nonelem() {
        let space = NdArraySpace::new(IntervalSpace::new(0.5, 1.5), ());
        assert!(!space.contains(&Array::zeros(())))
    }

    #[test]
    fn d2_empty_interval_contains_empty() {
        let space = NdArraySpace::new(IntervalSpace::new(0.5, 1.5), (2, 0));
        assert!(space.contains(&Array::zeros((2, 0))))
    }

    #[test]
    fn d2_interval_contains_inner_elems() {
        let space = NdArraySpace::new(IntervalSpace::new(0.5, 1.5), (2, 3));
        assert!(space.contains(&Array::linspace(0.9, 1.1, 6).into_shape((2, 3)).unwrap()))
    }

    #[test]
    fn d2_interval_not_contains_inner_mixed_elems() {
        let space = NdArraySpace::new(IntervalSpace::new(0.5, 1.5), (2, 3));
        assert!(!space.contains(&Array::linspace(0.0, 1.5, 6).into_shape((2, 3)).unwrap()))
    }

    #[test]
    fn d2_interval_not_contains_different_shape() {
        let space = NdArraySpace::new(IntervalSpace::new(0.5, 1.5), (2, 3));
        assert!(!space.contains(&Array::linspace(0.9, 1.1, 6).into_shape((3, 2)).unwrap()))
    }
    #[test]
    fn d0_singleton_interval_contains_samples() {
        let space = NdArraySpace::new(IntervalSpace::new(0.5_f32, 1.5), ());
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
            NdArraySpace::new(IntervalSpace::<f64>::default(), ()),
            NdArraySpace::new(IntervalSpace::<f64>::default(), ()),
        );
    }

    #[test]
    fn d2_interval_eq() {
        assert_eq!(
            NdArraySpace::new(IntervalSpace::<f64>::default(), (1, 2)),
            NdArraySpace::new(IntervalSpace::<f64>::default(), (1, 2)),
        );
    }

    #[test]
    fn d2_interval_ne_inner() {
        assert_ne!(
            NdArraySpace::new(IntervalSpace::<f64>::default(), (1, 2)),
            NdArraySpace::new(IntervalSpace::<f64>::new(0.0, 1.0), (1, 2)),
        );
    }

    #[test]
    fn d2_interval_ne_shapes() {
        assert_ne!(
            NdArraySpace::new(IntervalSpace::<f64>::default(), (1, 2)),
            NdArraySpace::new(IntervalSpace::<f64>::default(), (2, 1)),
        );
    }

    #[test]
    fn d2_interval_cmp_equal() {
        assert_eq!(
            NdArraySpace::new(IntervalSpace::<f64>::default(), (1, 2))
                .subset_cmp(&NdArraySpace::new(IntervalSpace::default(), (1, 2))),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn d2_interval_cmp_none_shapes() {
        assert_eq!(
            NdArraySpace::new(IntervalSpace::<f64>::default(), (1, 2))
                .subset_cmp(&NdArraySpace::new(IntervalSpace::default(), (2, 1))),
            None
        );
    }

    #[test]
    fn d2_interval_strict_subset() {
        assert!(NdArraySpace::new(IntervalSpace::new(0.0, 1.0), (1, 2))
            .strict_subset_of(&NdArraySpace::new(IntervalSpace::default(), (1, 2))));
    }
}

#[cfg(test)]
mod feature_space {
    use super::super::{BooleanSpace, IndexSpace, IntervalSpace};
    use super::*;

    #[test]
    fn d2_boolean_num_features() {
        let space = NdArraySpace::new(BooleanSpace, (2, 3));
        assert_eq!(space.num_features(), 6);
    }

    #[test]
    fn d2_index2_num_features() {
        let space = NdArraySpace::new(IndexSpace::new(2), (2, 3));
        assert_eq!(space.num_features(), 12);
    }

    features_tests!(
        d1_0_boolean,
        NdArraySpace::new(BooleanSpace, (0,)),
        Array::from_vec(vec![]),
        []
    );

    features_tests!(
        d1_2_boolean,
        NdArraySpace::new(BooleanSpace, (2,)),
        Array::from_vec(vec![true, false]),
        [1.0, 0.0]
    );

    features_tests!(
        d1_2_index2,
        NdArraySpace::new(IndexSpace::new(2), (2,)),
        Array::from_vec(vec![1, 0]),
        [0.0, 1.0, 1.0, 0.0]
    );

    features_tests!(
        d2_22_interval,
        NdArraySpace::new(IntervalSpace::new(0.0, 1.0), (2, 2)),
        Array::from_vec(vec![0.0, 0.25, 0.5, 0.75])
            .into_shape((2, 2))
            .unwrap(),
        [0.0, 0.25, 0.5, 0.75]
    );

    batch_features_tests!(
        batch_d1_0_index2,
        NdArraySpace::new(IndexSpace::new(2), (0,)),
        [Array::zeros((0,)), Array::zeros((0,))],
        [[], []]
    );

    batch_features_tests!(
        batch_d1_2_index2,
        NdArraySpace::new(IndexSpace::new(2), (2,)),
        [
            Array::from_vec(vec![0, 0]),
            Array::from_vec(vec![1, 1]),
            Array::from_vec(vec![1, 0]),
        ],
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    );

    batch_features_tests!(
        batch_d2_22_interval,
        NdArraySpace::new(IntervalSpace::new(0.0, 1.0), (2, 2)),
        [
            Array::from_vec(vec![0.0, 0.25, 0.5, 0.75])
                .into_shape((2, 2))
                .unwrap(),
            Array::from_vec(vec![0.1, 0.2, 0.3, 0.4])
                .into_shape((2, 2))
                .unwrap(),
        ],
        [[0.0, 0.25, 0.5, 0.75], [0.1, 0.2, 0.3, 0.4]]
    );
}
