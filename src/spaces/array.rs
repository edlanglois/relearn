//! Array space
use super::{
    iter_product_subset_ord, ElementRefInto, FeatureSpace, FiniteSpace, NonEmptySpace, Space,
    SubsetOrd,
};
use crate::logging::Loggable;
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::cmp::Ordering;

/// A Cartesian product of `N` spaces of the same type (but not necessarily the same space).
///
/// An `ArraySpace` is more general than a [`PowerSpace`](super::PowerSpace) because the inner
/// spaces do not all have to be the same, but less general than
/// a [tuple space](super::TupleSpace2) because the inner spaces must have the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
// See <https://github.com/est31/serde-big-array/issues/12#issue-1115462786>
#[serde(bound(
    serialize = "for<'a> S: Serialize + Deserialize<'a>",
    deserialize = "S: Serialize + Deserialize<'de>"
))]
pub struct ArraySpace<S, const N: usize> {
    // Serde does not natively support const generics, BigArray provides a work-around impl
    #[serde(with = "BigArray")]
    inner_spaces: [S; N],
}

impl<S, const N: usize> ArraySpace<S, N> {
    pub const fn new(inner_spaces: [S; N]) -> Self {
        Self { inner_spaces }
    }
}

impl<S: Space, const N: usize> Space for ArraySpace<S, N> {
    type Element = [S::Element; N];

    fn contains(&self, value: &Self::Element) -> bool {
        self.inner_spaces
            .iter()
            .zip(value)
            .all(|(s, v)| s.contains(v))
    }
}

impl<S: Space + SubsetOrd, const N: usize> SubsetOrd for ArraySpace<S, N> {
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        iter_product_subset_ord(
            self.inner_spaces
                .iter()
                .zip(&other.inner_spaces)
                .map(|(s, o)| s.subset_cmp(o)),
        )
    }
}

impl<S: FiniteSpace, const N: usize> FiniteSpace for ArraySpace<S, N> {
    fn size(&self) -> usize {
        self.inner_spaces
            .iter()
            .map(FiniteSpace::size)
            .fold(1, |accum, size| {
                accum
                    .checked_mul(size)
                    .expect("Size of space is larger than usize")
            })
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        // The index is obtained by treating the element as a little-endian number
        // when written as a sequence of inner-space indices.
        self.inner_spaces
            .iter()
            .rev()
            .zip(element.iter().rev())
            .fold(0, |index, (space, elem)| {
                index * space.size() + space.to_index(elem)
            })
    }

    fn from_index(&self, mut index: usize) -> Option<Self::Element> {
        let mut spaces_iter = self.inner_spaces.iter();

        let result_elems = array_init::try_array_init(|_| {
            let space = spaces_iter.next().unwrap();
            let size = space.size();
            let result_elem = space.from_index(index % size).ok_or(());
            index /= size;
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

impl<S: NonEmptySpace, const N: usize> NonEmptySpace for ArraySpace<S, N> {
    fn some_element(&self) -> <Self as Space>::Element {
        array_init::array_init(|i| self.inner_spaces[i].some_element())
    }
}

impl<S, const N: usize> Distribution<<Self as Space>::Element> for ArraySpace<S, N>
where
    S: Space + Distribution<S::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        array_init::array_init(|i| self.inner_spaces[i].sample(rng))
    }
}

/// Features are the concatenation of inner feature vectors
impl<S: FeatureSpace, const N: usize> FeatureSpace for ArraySpace<S, N> {
    #[inline]
    fn num_features(&self) -> usize {
        self.inner_spaces
            .iter()
            .map(FeatureSpace::num_features)
            .sum()
    }

    #[inline]
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        zeroed: bool,
    ) -> &'a mut [F] {
        self.inner_spaces
            .iter()
            .zip(element)
            .fold(out, |out, (inner_space, inner_elem)| {
                inner_space.features_out(inner_elem, out, zeroed)
            })
    }
}

impl<S: Space, const N: usize> ElementRefInto<Loggable> for ArraySpace<S, N> {
    fn elem_ref_into(&self, _element: &Self::Element) -> Loggable {
        // Too complex to log
        Loggable::Nothing
    }
}

#[cfg(test)]
mod space {
    use super::super::{testing, IndexSpace};
    use super::*;

    #[test]
    fn empty_contains_empty() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        assert!(space.contains(&[]));
    }

    #[test]
    fn empty_contains_samples() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn i3_contains_2() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        assert!(space.contains(&[2]));
    }

    #[test]
    fn i3_not_contains_4() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        assert!(!space.contains(&[4]));
    }

    #[test]
    fn i3_contains_samples() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn i3i4_contains_2_0() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert!(space.contains(&[2, 1]));
    }

    #[test]
    fn i3i4_not_contains_2_4() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert!(!space.contains(&[2, 4]));
    }

    #[test]
    fn i3i4_contains_samples() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        testing::check_contains_samples(&space, 10);
    }
}

#[cfg(test)]
mod subset_ord {
    use super::super::IndexSpace;
    use super::*;

    #[test]
    fn i3i4_eq_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Equal));
        assert_eq!(s1, s2);
        assert!(s1.subset_of(&s2));
        assert!(!s1.strict_subset_of(&s2));
        assert!(s1.superset_of(&s2));
        assert!(!s1.strict_superset_of(&s2));
    }

    #[test]
    fn i3i4_ne_i4i3() {
        let s1 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(4), IndexSpace::new(3)]);
        assert!(s1.subset_cmp(&s2).is_none());
        assert_ne!(s1, s2);
        assert!(!s1.subset_of(&s2));
        assert!(!s1.strict_subset_of(&s2));
        assert!(!s1.superset_of(&s2));
        assert!(!s1.strict_superset_of(&s2));
    }

    #[test]
    fn i2i4_lt_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(2), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Less));
        assert!(s1.subset_of(&s2));
        assert!(s1.strict_subset_of(&s2));
        assert!(!s1.superset_of(&s2));
        assert!(!s1.strict_superset_of(&s2));
    }

    #[test]
    fn i3i3_lt_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(3)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Less));
        assert!(s1.subset_of(&s2));
        assert!(s1.strict_subset_of(&s2));
        assert!(!s1.superset_of(&s2));
        assert!(!s1.strict_superset_of(&s2));
    }

    #[test]
    fn i5i4_gt_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(5), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(s1.subset_cmp(&s2), Some(Ordering::Greater));
        assert!(!s1.subset_of(&s2));
        assert!(!s1.strict_subset_of(&s2));
        assert!(s1.superset_of(&s2));
        assert!(s1.strict_superset_of(&s2));
    }

    #[test]
    fn i2i4_incomp_i3i3() {
        let s1 = ArraySpace::new([IndexSpace::new(2), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(3)]);
        assert!(s1.subset_cmp(&s2).is_none());
        assert_ne!(s1, s2);
        assert!(!s1.subset_of(&s2));
        assert!(!s1.strict_subset_of(&s2));
        assert!(!s1.superset_of(&s2));
        assert!(!s1.strict_superset_of(&s2));
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::{testing, IndexSpace};
    use super::*;

    #[test]
    fn empty_size() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        assert_eq!(space.size(), 1);
    }
    #[test]
    fn empty_to_index() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        assert_eq!(space.to_index(&[]), 0);
    }
    #[test]
    fn empty_from_index() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        assert_eq!(space.from_index(0), Some([]));
    }
    #[test]
    fn empty_to_from_index_iter_size() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        testing::check_from_to_index_iter_size(&space);
    }
    #[test]
    fn empty_from_to_index_random() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        testing::check_from_to_index_random(&space, 10);
    }
    #[test]
    fn empty_from_index_sampled() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        testing::check_from_index_sampled(&space, 10);
    }
    #[test]
    fn empty_from_index_invalid() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn i3_size() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        assert_eq!(space.size(), 3);
    }
    #[test]
    fn i3_to_index() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        assert_eq!(space.to_index(&[1]), 1);
    }
    #[test]
    fn i3_from_index() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        assert_eq!(space.from_index(1), Some([1]));
    }
    #[test]
    fn i3_to_from_index_iter_size() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        testing::check_from_to_index_iter_size(&space);
    }
    #[test]
    fn i3_from_to_index_random() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        testing::check_from_to_index_random(&space, 10);
    }
    #[test]
    fn i3_from_index_sampled() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        testing::check_from_index_sampled(&space, 10);
    }
    #[test]
    fn i3_from_index_invalid() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn i3i4_size() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(space.size(), 12);
    }
    #[test]
    fn i3i4_to_index() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(space.to_index(&[1, 2]), 7);
    }
    #[test]
    fn i3i4_from_index() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(space.from_index(7), Some([1, 2]));
    }
    #[test]
    fn i3i4_to_from_index_iter_size() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        testing::check_from_to_index_iter_size(&space);
    }
    #[test]
    fn i3i4_from_to_index_random() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        testing::check_from_to_index_random(&space, 10);
    }
    #[test]
    fn i3i4_from_index_sampled() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        testing::check_from_index_sampled(&space, 10);
    }
    #[test]
    fn i3i4_from_index_invalid() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        testing::check_from_index_invalid(&space);
    }
}

#[cfg(test)]
mod feature_space {
    use super::super::IndexSpace;
    use super::*;

    mod empty {
        use super::*;

        const fn space() -> ArraySpace<IndexSpace, 0> {
            ArraySpace::new([])
        }

        #[test]
        fn num_features() {
            let space = space();
            assert_eq!(space.num_features(), 0);
        }
        features_tests!(f, space(), [], []);
        batch_features_tests!(b, space(), [[], [], []], [[], [], []]);
    }

    #[test]
    fn i3_num_features() {
        let space = ArraySpace::new([IndexSpace::new(3)]);
        assert_eq!(space.num_features(), 3);
    }

    #[test]
    fn i3i4_num_features() {
        let space = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(space.num_features(), 7);
    }

    features_tests!(
        i3i4,
        ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]),
        [0, 0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    );
    features_tests!(
        i3i4_2,
        ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]),
        [1, 3],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );

    batch_features_tests!(
        batch_i2i3,
        ArraySpace::new([IndexSpace::new(2), IndexSpace::new(3)]),
        [[0, 0], [1, 2]],
        [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]]
    );
}
