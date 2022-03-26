//! `BooleanSpace` definition
use super::{
    ElementRefInto, FeatureSpace, FiniteSpace, NonEmptySpace, ParameterizedDistributionSpace,
    ReprSpace, Space, SubsetOrd,
};
use crate::logging::Loggable;
use crate::torch::distributions::Bernoulli;
use crate::utils::distributions::ArrayDistribution;
use crate::utils::num_array::{BuildFromArray1D, NumArray1D};
use num_traits::Float;
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use tch::{Device, Kind, Tensor};

/// The space `{false, true}`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BooleanSpace;

impl BooleanSpace {
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        BooleanSpace
    }
}

impl fmt::Display for BooleanSpace {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BooleanSpace")
    }
}

impl Space for BooleanSpace {
    type Element = bool;

    #[inline]
    fn contains(&self, _value: &Self::Element) -> bool {
        true
    }
}

impl SubsetOrd for BooleanSpace {
    #[inline]
    fn subset_cmp(&self, _other: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl FiniteSpace for BooleanSpace {
    #[inline]
    fn size(&self) -> usize {
        2
    }

    #[inline]
    fn to_index(&self, element: &Self::Element) -> usize {
        (*element).into()
    }

    #[inline]
    fn from_index(&self, index: usize) -> Option<Self::Element> {
        match index {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    #[inline]
    fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
        Some(index != 0)
    }
}

impl NonEmptySpace for BooleanSpace {
    #[inline]
    fn some_element(&self) -> Self::Element {
        false
    }
}

/// Represent elements as a Boolean valued tensor.
impl ReprSpace<Tensor> for BooleanSpace {
    #[inline]
    fn repr(&self, element: &Self::Element) -> Tensor {
        Tensor::scalar_tensor(i64::from(*element), (Kind::Bool, Device::Cpu))
    }

    #[inline]
    fn batch_repr<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        let elements: Vec<_> = elements.into_iter().copied().collect();
        Tensor::of_slice(&elements)
    }
}

impl ParameterizedDistributionSpace<Tensor> for BooleanSpace {
    type Distribution = Bernoulli;

    #[inline]
    fn num_distribution_params(&self) -> usize {
        1
    }

    #[inline]
    fn sample_element(&self, params: &Tensor) -> Self::Element {
        self.distribution(params).sample().into()
    }

    #[inline]
    fn distribution(&self, params: &Tensor) -> Self::Distribution {
        Self::Distribution::new(params.squeeze_dim(-1))
    }
}

/// Features are `[0.0]` for `false` and `[1.0]` for `true`
impl FeatureSpace for BooleanSpace {
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
        out[0] = if *element { F::one() } else { F::zero() };
        &mut out[1..]
    }

    #[inline]
    fn features<T>(&self, element: &Self::Element) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        if *element {
            T::Array::ones(1).into()
        } else {
            T::Array::zeros(1).into()
        }
    }
}

impl Distribution<<Self as Space>::Element> for BooleanSpace {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        rng.gen()
    }
}

impl ElementRefInto<Loggable> for BooleanSpace {
    #[inline]
    fn elem_ref_into(&self, element: &Self::Element) -> Loggable {
        Loggable::Scalar(u8::from(*element).into())
    }
}

#[cfg(test)]
mod space {
    use super::super::testing;
    use super::*;

    #[test]
    fn contains_false() {
        let space = BooleanSpace;
        assert!(space.contains(&false));
    }

    #[test]
    fn contains_true() {
        let space = BooleanSpace;
        assert!(space.contains(&true));
    }

    #[test]
    fn contains_samples() {
        let space = BooleanSpace;
        testing::check_contains_samples(&space, 10);
    }
}

#[cfg(test)]
mod subset_ord {
    use super::*;

    #[test]
    fn eq() {
        assert_eq!(BooleanSpace, BooleanSpace);
    }

    #[test]
    fn cmp_equal() {
        assert_eq!(
            BooleanSpace.subset_cmp(&BooleanSpace),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn not_less() {
        assert!(!BooleanSpace.strict_subset_of(&BooleanSpace));
    }
}

#[cfg(test)]
mod finite_space {
    use super::super::testing;
    use super::*;

    #[test]
    fn from_to_index_iter_size() {
        let space = BooleanSpace;
        testing::check_from_to_index_iter_size(&space);
    }

    #[test]
    fn from_to_index_random() {
        let space = BooleanSpace;
        testing::check_from_to_index_random(&space, 10);
    }

    #[test]
    fn from_index_sampled() {
        let space = BooleanSpace;
        testing::check_from_index_sampled(&space, 10);
    }

    #[test]
    fn from_index_invalid() {
        let space = BooleanSpace;
        testing::check_from_index_invalid(&space);
    }
}

#[cfg(test)]
mod feature_space {
    use super::*;

    #[test]
    fn num_features() {
        assert_eq!(BooleanSpace.num_features(), 1);
    }

    features_tests!(false_, BooleanSpace, false, [0.0]);
    features_tests!(true_, BooleanSpace, true, [1.0]);
    batch_features_tests!(
        batch,
        BooleanSpace,
        [false, true, true, false],
        [[0.0], [1.0], [1.0], [0.0]]
    );
}

#[cfg(test)]
mod parameterized_sample_space_tensor {
    use super::*;
    use std::iter;

    #[test]
    fn num_sample_params() {
        assert_eq!(1, BooleanSpace.num_distribution_params());
    }

    #[test]
    fn sample_element_deterministic() {
        let space = BooleanSpace;
        let params = Tensor::of_slice(&[f32::INFINITY]);
        for _ in 0..10 {
            assert!(space.sample_element(&params));
        }
    }

    #[test]
    fn sample_element_check_distribution() {
        let space = BooleanSpace;
        // logit = 1.0; p ~= 0.731
        let params = Tensor::of_slice(&[1.0f32]);
        let p = 0.731;
        let n = 5000;
        let count: u64 = iter::repeat_with(|| if space.sample_element(&params) { 1 } else { 0 })
            .take(n)
            .sum();
        // Check that the counts are within a confidence interval
        // Using Wald method <https://en.wikipedia.org/wiki/Binomial_distribution#Wald_method>
        // Quantile for error rate of 1e-5
        let z = 4.4;
        let nf = n as f64;
        let stddev = (p * (1.0 - p) * nf).sqrt();
        let lower_bound = nf * p - z * stddev; // ~717
        let upper_bound = nf * p + z * stddev; // ~745
        assert!(lower_bound <= count as f64);
        assert!(upper_bound >= count as f64);
    }
}
