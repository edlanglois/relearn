//! An array of N copies of an inner space.
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, ElementRefInto, FeatureSpace,
    FeatureSpaceOut, FiniteSpace, Space,
};
use crate::logging::Loggable;
use rand::distributions::Distribution;
use rand::Rng;
use std::cmp::Ordering;
use tch::{Device, Kind, Tensor};

/// A Cartesian product of n copies of the same space.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArraySpace<S, const N: usize> {
    pub inner_space: S,
}

impl<S, const N: usize> ArraySpace<S, N> {
    pub const fn new(inner_space: S) -> Self {
        Self { inner_space }
    }
}

impl<S: Space, const N: usize> Space for ArraySpace<S, N> {
    type Element = [S::Element; N];

    fn contains(&self, value: &Self::Element) -> bool {
        value.iter().all(|v| self.inner_space.contains(v))
    }
}

impl<S: Space + PartialOrd, const N: usize> PartialOrd for ArraySpace<S, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner_space.partial_cmp(&other.inner_space)
    }
}

impl<S: FiniteSpace, const N: usize> FiniteSpace for ArraySpace<S, N> {
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
        let result = array_init::try_array_init(|_| {
            let result = self.inner_space.from_index(index % inner_size).ok_or(());
            index /= inner_size;
            result
        })
        .ok();
        if index == 0 {
            result
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
        array_init::array_init(|_| self.inner_space.sample(rng))
    }
}

impl<S: BaseFeatureSpace, const N: usize> BaseFeatureSpace for ArraySpace<S, N> {
    fn num_features(&self) -> usize {
        self.inner_space.num_features() * N
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
        let split_size = self.inner_space.num_features().try_into().unwrap();
        for (inner_tensor, inner_elem) in out.split(split_size, -1).iter_mut().zip(element.iter()) {
            self.inner_space
                .features_out(inner_elem, inner_tensor, zeroed);
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
            self.inner_space
                .batch_features_out(elements.into_iter().map(|e| &e[0]), out, zeroed);
            return;
        }
        let elements_iter = elements.into_iter();
        for (i, view) in out.tensor_split(N as i64, -1).iter_mut().enumerate() {
            self.inner_space
                .batch_features_out(elements_iter.clone().map(|e| &e[i]), view, zeroed)
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
    use super::super::{testing, BooleanSpace, IntervalSpace};
    use super::*;

    #[test]
    fn d0_boolean_contains_empty() {
        let space = ArraySpace::<_, 0>::new(BooleanSpace);
        assert!(space.contains(&[]));
    }

    #[test]
    fn d1_boolean_contains_true() {
        let space = ArraySpace::<_, 1>::new(BooleanSpace);
        assert!(space.contains(&[true]));
    }

    #[test]
    fn d2_boolean_contains_true_false() {
        let space = ArraySpace::<_, 2>::new(BooleanSpace);
        assert!(space.contains(&[true, false]));
    }

    #[test]
    fn d2_interval_contains_point() {
        let space = ArraySpace::<_, 2>::new(IntervalSpace::default());
        assert!(space.contains(&[3.0, -100.0]));
    }

    #[test]
    fn d2_unit_box_not_contains_point() {
        let space = ArraySpace::<_, 2>::new(IntervalSpace::new(0.0, 1.0));
        assert!(!space.contains(&[0.5, 2.1]));
    }

    #[test]
    fn d0_boolean_contains_samples() {
        let space = ArraySpace::<_, 0>::new(BooleanSpace);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn d1_boolean_contains_samples() {
        let space = ArraySpace::<_, 1>::new(BooleanSpace);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn d2_boolean_contains_samples() {
        let space = ArraySpace::<_, 2>::new(BooleanSpace);
        testing::check_contains_samples(&space, 10);
    }

    #[test]
    fn d2_interval_contains_samples() {
        let space = ArraySpace::<_, 2>::new(IntervalSpace::<f64>::default());
        testing::check_contains_samples(&space, 10);
    }
}

#[cfg(test)]
mod partial_ord {
    use super::super::IntervalSpace;
    use super::*;

    #[test]
    fn d0_interval_eq() {
        assert_eq!(
            ArraySpace::<_, 0>::new(IntervalSpace::<f64>::default()),
            ArraySpace::<_, 0>::new(IntervalSpace::<f64>::default())
        );
    }

    #[test]
    fn d2_interval_eq() {
        assert_eq!(
            ArraySpace::<_, 2>::new(IntervalSpace::<f64>::default()),
            ArraySpace::<_, 2>::new(IntervalSpace::<f64>::default())
        );
    }

    #[test]
    fn d2_interval_ne() {
        assert_ne!(
            ArraySpace::<_, 2>::new(IntervalSpace::default()),
            ArraySpace::<_, 2>::new(IntervalSpace::new(0.0, 1.0))
        );
    }

    #[test]
    fn d2_interval_cmp_equal() {
        assert_eq!(
            ArraySpace::<_, 2>::new(IntervalSpace::<f64>::default())
                .partial_cmp(&ArraySpace::new(IntervalSpace::default())),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn d2_interval_lt() {
        assert!(
            ArraySpace::<_, 2>::new(IntervalSpace::new(0.0, 1.0))
                < ArraySpace::<_, 2>::new(IntervalSpace::default())
        );
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::{testing, BooleanSpace};
    use super::*;

    #[test]
    fn d3_boolean_from_to_index_iter_size() {
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn d3_boolean_from_to_index_random() {
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
        testing::check_from_to_index_random(&space, 10);
    }

    #[test]
    fn d3_boolean_from_index_sampled() {
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
        testing::check_from_index_sampled(&space, 10);
    }

    #[test]
    fn d3_boolean_from_index_invalid() {
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
        testing::check_from_index_invalid(&space);
    }

    #[test]
    fn d0_boolean_size() {
        let space = ArraySpace::<_, 0>::new(BooleanSpace);
        assert_eq!(space.size(), 1);
    }

    #[test]
    fn d0_boolean_to_index() {
        let space = ArraySpace::<_, 0>::new(BooleanSpace);
        assert_eq!(space.to_index(&[]), 0);
    }

    #[test]
    fn d0_boolean_from_index() {
        let space = ArraySpace::<_, 0>::new(BooleanSpace);
        assert_eq!(space.from_index(0), Some([]));
    }

    #[test]
    fn d3_boolean_size() {
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
        assert_eq!(space.size(), 8);
    }

    #[test]
    fn d3_boolean_to_index() {
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
        assert_eq!(space.to_index(&[false, false, false]), 0);
        assert_eq!(space.to_index(&[true, false, false]), 1);
        assert_eq!(space.to_index(&[false, true, false]), 2);
        assert_eq!(space.to_index(&[false, false, true]), 4);
        assert_eq!(space.to_index(&[true, true, true]), 7);
    }

    #[test]
    fn d3_boolean_from_index() {
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
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
        let space = ArraySpace::<_, 3>::new(BooleanSpace);
        assert_eq!(space.num_features(), 3);
    }

    #[test]
    fn d3_index2_num_features() {
        let space = ArraySpace::<_, 3>::new(IndexSpace::new(2));
        assert_eq!(space.num_features(), 6);
    }

    macro_rules! features_tests {
        ($label:ident, $inner:expr, $n:literal, $elem:expr, $expected:expr) => {
            mod $label {
                use super::*;

                #[test]
                fn tensor_features() {
                    let space = ArraySpace::<_, $n>::new($inner);
                    let actual: Tensor = space.features(&$elem);
                    let expected_vec: &[f32] = &$expected;
                    assert_eq!(actual, Tensor::of_slice(expected_vec));
                }

                #[test]
                fn tensor_features_out() {
                    let space = ArraySpace::<_, $n>::new($inner);
                    let expected_vec: &[f32] = &$expected;
                    let expected = Tensor::of_slice(&expected_vec);
                    let mut out = expected.empty_like();
                    space.features_out(&$elem, &mut out, false);
                    assert_eq!(out, expected);
                }
            }
        };
    }

    features_tests!(d0_boolean_, BooleanSpace, 0, [], []);
    features_tests!(d2_boolean_, BooleanSpace, 2, [true, false], [1.0, 0.0]);
    features_tests!(
        d2_index2_,
        IndexSpace::new(2),
        2,
        [1, 0],
        [0.0, 1.0, 1.0, 0.0]
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
    fn d2_index2_tensor_batch_features() {
        let space = ArraySpace::<_, 2>::new(IndexSpace::new(2));
        let actual: Tensor = space.batch_features(&[[0, 0], [1, 1], [1, 0]]);
        let expected = tensor_from_arrays([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn d0_index2_tensor_batch_features_out() {
        let space = ArraySpace::<_, 0>::new(IndexSpace::new(2));
        let expected = tensor_from_arrays([[], []]);
        let mut out = expected.empty_like();
        space.batch_features_out(&[[], []], &mut out, false);
        assert_eq!(out, expected);
    }

    #[test]
    fn d1_index2_tensor_batch_features_out() {
        let space = ArraySpace::<_, 1>::new(IndexSpace::new(2));
        let expected = tensor_from_arrays([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]);
        let mut out = expected.empty_like();
        space.batch_features_out(&[[0], [1], [1]], &mut out, false);
        assert_eq!(out, expected);
    }

    #[test]
    fn d2_index2_tensor_batch_features_out() {
        let space = ArraySpace::<_, 2>::new(IndexSpace::new(2));
        let expected = tensor_from_arrays([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]);
        let mut out = expected.empty_like();
        space.batch_features_out(&[[0, 0], [1, 1], [1, 0]], &mut out, false);
        assert_eq!(out, expected);
    }
}
