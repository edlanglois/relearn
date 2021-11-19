//! Cartesian power space.
use super::{ElementRefInto, EncoderFeatureSpace, FiniteSpace, NumFeatures, Space, SubsetOrd};
use crate::logging::Loggable;
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use std::cmp::Ordering;

/// A Cartesian power of a space: a Cartesian product of `N` copies of the same space.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PowerSpace<S, const N: usize> {
    pub inner_space: S,
}

impl<S, const N: usize> PowerSpace<S, N> {
    pub const fn new(inner_space: S) -> Self {
        Self { inner_space }
    }
}

impl<S: Space, const N: usize> Space for PowerSpace<S, N> {
    type Element = [S::Element; N];

    fn contains(&self, value: &Self::Element) -> bool {
        value.iter().all(|v| self.inner_space.contains(v))
    }
}

impl<S: SubsetOrd, const N: usize> SubsetOrd for PowerSpace<S, N> {
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner_space.subset_cmp(&other.inner_space)
    }
}

impl<S: FiniteSpace, const N: usize> FiniteSpace for PowerSpace<S, N> {
    fn size(&self) -> usize {
        self.inner_space
            .size()
            .checked_pow(N.try_into().expect("Size of space is larger than usize"))
            .expect("Size of space is larger than usize")
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        // The index is obtained by treating the element as a little-endian number
        // when written as a sequence of inner-space indices.
        let inner_size = self.inner_space.size();
        let mut index = 0;
        for inner_elem in element.iter().rev() {
            index *= inner_size;
            index += self.inner_space.to_index(inner_elem)
        }
        index
    }

    fn from_index(&self, mut index: usize) -> Option<Self::Element> {
        let inner_size = self.inner_space.size();
        let result_elems = array_init::try_array_init(|_| {
            let result_elem = self.inner_space.from_index(index % inner_size).ok_or(());
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

impl<S, const N: usize> Distribution<<Self as Space>::Element> for PowerSpace<S, N>
where
    S: Space + Distribution<S::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        array_init::array_init(|_| self.inner_space.sample(rng))
    }
}

impl<S: NumFeatures, const N: usize> NumFeatures for PowerSpace<S, N> {
    fn num_features(&self) -> usize {
        self.inner_space.num_features() * N
    }
}

/// Feature encoder for [`PowerSpace`].
pub struct PowerSpaceEncoder<T> {
    inner_encoder: T,
    inner_num_features: usize,
}

impl<S: EncoderFeatureSpace, const N: usize> EncoderFeatureSpace for PowerSpace<S, N> {
    type Encoder = PowerSpaceEncoder<S::Encoder>;

    fn encoder(&self) -> Self::Encoder {
        PowerSpaceEncoder {
            inner_encoder: self.inner_space.encoder(),
            inner_num_features: self.inner_space.num_features(),
        }
    }

    fn encoder_features_out<F: Float>(
        &self,
        element: &Self::Element,
        out: &mut [F],
        zeroed: bool,
        encoder: &Self::Encoder,
    ) {
        let mut chunks = out.chunks_exact_mut(encoder.inner_num_features);
        for inner_elem in element {
            let chunk = chunks.next().expect("output slice is too small");
            self.inner_space.encoder_features_out(
                inner_elem,
                chunk,
                zeroed,
                &encoder.inner_encoder,
            );
        }
    }
}

impl<S: Space, const N: usize> ElementRefInto<Loggable> for PowerSpace<S, N> {
    fn elem_ref_into(&self, _element: &Self::Element) -> Loggable {
        // Too complex to log
        Loggable::Nothing
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
