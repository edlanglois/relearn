//! Array space
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, ElementRefInto, FeatureSpace,
    FeatureSpaceOut, FiniteSpace, Space,
};
use crate::logging::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
use std::cmp::Ordering;
use tch::{Device, Kind, Tensor};

/// A Cartesian product of `N` spaces of the same type (but not necessarily the same space).
///
/// An `ArraySpace` is more general than a [`PowerSpace`](super::PowerSpace) because the inner
/// spaces do not all have to be the same, but less general than
/// a [`ProductSpace`](super::ProductSpace) because the inner spaces must have the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArraySpace<S, const N: usize> {
    pub inner_spaces: [S; N],
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

impl<S: Space + PartialOrd, const N: usize> PartialOrd for ArraySpace<S, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner_spaces
            .iter()
            .zip(&other.inner_spaces)
            .map(|(s, o)| s.partial_cmp(o))
            .try_fold(Ordering::Equal, |state, cmp| match cmp {
                None => None,
                Some(Ordering::Equal) => Some(state),
                Some(Ordering::Less) => {
                    if state == Ordering::Greater {
                        None
                    } else {
                        Some(Ordering::Less)
                    }
                }
                Some(Ordering::Greater) => {
                    if state == Ordering::Less {
                        None
                    } else {
                        Some(Ordering::Greater)
                    }
                }
            })
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

impl<S, const N: usize> Distribution<<Self as Space>::Element> for ArraySpace<S, N>
where
    S: Space + Distribution<S::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        array_init::from_iter(self.inner_spaces.iter().map(|s| s.sample(rng))).unwrap()
    }
}

impl<S: BaseFeatureSpace, const N: usize> BaseFeatureSpace for ArraySpace<S, N> {
    fn num_features(&self) -> usize {
        self.inner_spaces
            .iter()
            .map(BaseFeatureSpace::num_features)
            .sum()
    }
}

impl<S: FeatureSpaceOut<Tensor>, const N: usize> FeatureSpace<Tensor> for ArraySpace<S, N> {
    fn features(&self, element: &Self::Element) -> Tensor {
        let mut out = Tensor::empty(&[self.num_features() as i64], (Kind::Float, Device::Cpu));
        self.features_out(element, &mut out, false);
        out
    }
}

impl<S: FeatureSpaceOut<Tensor>, const N: usize> FeatureSpaceOut<Tensor> for ArraySpace<S, N> {
    fn features_out(&self, element: &Self::Element, out: &mut Tensor, zeroed: bool) {
        if N == 0 {
            return;
        } else if N == 1 {
            self.inner_spaces[0].features_out(&element[0], out, zeroed);
            return;
        }

        let split_sizes: Vec<i64> = self
            .inner_spaces
            .iter()
            .map(|s| s.num_features().try_into().unwrap())
            .collect();
        for (inner_tensor, (inner_space, inner_elem)) in out
            .split_with_sizes(&split_sizes, -1)
            .iter_mut()
            .zip(self.inner_spaces.iter().zip(element))
        {
            inner_space.features_out(inner_elem, inner_tensor, zeroed);
        }
    }
}

impl<S, const N: usize> BatchFeatureSpace<Tensor> for ArraySpace<S, N>
where
    S: BatchFeatureSpaceOut<Tensor>,
    S::Element: 'static,
{
    fn batch_features<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        let elements_iter = elements.into_iter();
        let mut out = Tensor::empty(
            &[elements_iter.len() as i64, self.num_features() as i64],
            (Kind::Float, Device::Cpu),
        );
        self.batch_features_out(elements_iter, &mut out, false);
        out
    }
}

impl<S, const N: usize> BatchFeatureSpaceOut<Tensor> for ArraySpace<S, N>
where
    S: BatchFeatureSpaceOut<Tensor>,
    S::Element: 'static,
{
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut Tensor, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: Clone,
        Self::Element: 'a,
    {
        if N == 0 {
            return;
        } else if N == 1 {
            self.inner_spaces[0].batch_features_out(
                elements.into_iter().map(|e| &e[0]),
                out,
                zeroed,
            );
            return;
        }
        let elements_iter = elements.into_iter();
        let split_sizes: Vec<i64> = self
            .inner_spaces
            .iter()
            .map(|s| s.num_features().try_into().unwrap())
            .collect();
        for (i, (inner_tensor, inner_space)) in out
            .split_with_sizes(&split_sizes, -1)
            .iter_mut()
            .zip(&self.inner_spaces)
            .enumerate()
        {
            inner_space.batch_features_out(
                elements_iter.clone().map(|e| &e[i]),
                inner_tensor,
                zeroed,
            )
        }
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
mod partial_ord {
    use super::super::IndexSpace;
    use super::*;

    #[test]
    fn i3i4_eq_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert_eq!(s1, s2);
    }

    #[test]
    fn i3i4_ne_i4i3() {
        let s1 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(4), IndexSpace::new(3)]);
        assert_ne!(s1, s2);
    }

    #[test]
    fn i2i4_lt_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(2), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert!(s1 < s2);
    }

    #[test]
    fn i3i3_lt_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(3)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert!(s1 < s2);
    }

    #[test]
    fn i5i4_gt_i3i4() {
        let s1 = ArraySpace::new([IndexSpace::new(5), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(4)]);
        assert!(s1 > s2);
    }

    #[test]
    fn i2i4_incomp_i3i3() {
        let s1 = ArraySpace::new([IndexSpace::new(2), IndexSpace::new(4)]);
        let s2 = ArraySpace::new([IndexSpace::new(3), IndexSpace::new(3)]);
        assert!(s1.partial_cmp(&s2).is_none());
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

    #[test]
    fn empty_num_features() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        assert_eq!(space.num_features(), 0);
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

    macro_rules! features_tests {
        ($label:ident, $space:expr, $elem:expr, $expected:expr) => {
            mod $label {
                use super::*;

                #[test]
                fn tensor_features() {
                    let space = $space;
                    let actual: Tensor = space.features(&$elem);
                    let expected_vec: &[f32] = &$expected;
                    assert_eq!(actual, Tensor::of_slice(expected_vec));
                }

                #[test]
                fn tensor_features_out() {
                    let space = $space;
                    let expected_vec: &[f32] = &$expected;
                    let expected = Tensor::of_slice(&expected_vec);
                    let mut out = expected.empty_like();
                    space.features_out(&$elem, &mut out, false);
                    assert_eq!(out, expected);
                }
            }
        };
    }

    features_tests!(empty, ArraySpace::<IndexSpace, 0>::new([]), [], []);
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

    fn tensor_from_arrays<const N: usize, const M: usize>(data: [[f32; M]; N]) -> Tensor {
        let flat_data: Vec<f32> = data
            .into_iter()
            .map(IntoIterator::into_iter)
            .flatten()
            .collect();
        Tensor::of_slice(&flat_data).reshape(&[N as i64, M as i64])
    }

    #[test]
    fn empty_tensor_batch_features() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        let actual: Tensor = space.batch_features(&[[], [], []]);
        let expected = tensor_from_arrays([[], [], []]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_tensor_batch_features_out() {
        let space = ArraySpace::<IndexSpace, 0>::new([]);
        let expected = tensor_from_arrays([[], [], []]);
        let mut out = expected.empty_like();
        space.batch_features_out(&[[], [], []], &mut out, false);
        assert_eq!(out, expected);
    }

    #[test]
    fn i2i3_tensor_batch_features() {
        let space = ArraySpace::new([IndexSpace::new(2), IndexSpace::new(3)]);
        let actual: Tensor = space.batch_features(&[[0, 0], [1, 2]]);
        let expected = tensor_from_arrays([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn i3i4_tensor_batch_features_out() {
        let space = ArraySpace::new([IndexSpace::new(2), IndexSpace::new(3)]);
        let expected = tensor_from_arrays([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]]);
        let mut out = expected.empty_like();
        space.batch_features_out(&[[0, 0], [1, 2]], &mut out, false);
        assert_eq!(out, expected);
    }
}
