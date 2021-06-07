//! Wrap spaces to have non-empty feature vectors.
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, FeatureSpace, FeatureSpaceOut,
    FiniteSpace, Space,
};
use crate::utils::array::{BasicArray, BasicArrayMut};
use std::fmt;

// Note: This could be renamed to something like TorchSpace if other helper changes end up being
// requried to satisfy torch methods.

/// Wrapper space with a feature vector length of at least 1.
///
/// Any generated features have value `0`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

impl<S: BaseFeatureSpace> BaseFeatureSpace for NonEmptyFeatures<S> {
    fn num_features(&self) -> usize {
        self.inner.num_features().max(1)
    }
}

impl<S, T> FeatureSpace<T> for NonEmptyFeatures<S>
where
    S: FeatureSpace<T>,
    T: BasicArray<f32, 1>,
{
    fn features(&self, element: &Self::Element) -> T {
        if self.inner.num_features() == 0 {
            T::zeros([1])
        } else {
            self.inner.features(element)
        }
    }
}

impl<S, T2> BatchFeatureSpace<T2> for NonEmptyFeatures<S>
where
    S: BatchFeatureSpace<T2>,
    T2: BasicArray<f32, 2>,
{
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
    {
        if self.inner.num_features() == 0 {
            let num_elements = elements.into_iter().count();
            T2::zeros([num_elements, 1])
        } else {
            self.inner.batch_features(elements)
        }
    }
}

impl<S, T> FeatureSpaceOut<T> for NonEmptyFeatures<S>
where
    S: FeatureSpaceOut<T>,
    T: BasicArrayMut,
{
    fn features_out(&self, element: &Self::Element, out: &mut T, zeroed: bool) {
        if self.inner.num_features() == 0 {
            if !zeroed {
                out.zero_();
            }
        } else {
            self.inner.features_out(element, out, zeroed);
        }
    }
}

impl<S, T2> BatchFeatureSpaceOut<T2> for NonEmptyFeatures<S>
where
    S: BatchFeatureSpaceOut<T2>,
    T2: BasicArrayMut,
{
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut T2, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        if self.inner.num_features() == 0 {
            if !zeroed {
                out.zero_();
            }
        } else {
            self.inner.batch_features_out(elements, out, zeroed);
        }
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

    macro_rules! tests {
        ($array:ty, $zeros:expr, $empty:expr) => {
            #[test]
            fn features_wrap_0() {
                let space = NonEmptyFeatures::new(SingletonSpace::new());
                let actual: $array = space.features(&());
                assert_eq!(actual, $zeros([1]));
            }

            #[test]
            fn features_wrap_1() {
                let inner = IndexSpace::new(1);
                let space = NonEmptyFeatures::new(inner.clone());
                let actual: $array = space.features(&0);
                let expected: $array = inner.features(&0);
                assert_eq!(actual, expected);
            }

            #[test]
            fn features_wrap_2() {
                let inner = IndexSpace::new(2);
                let space = NonEmptyFeatures::new(inner.clone());
                let actual: $array = space.features(&1);
                let expected: $array = inner.features(&1);
                assert_eq!(actual, expected);
            }

            #[test]
            fn features_out_wrap_0() {
                let space = NonEmptyFeatures::new(SingletonSpace::new());
                let mut out = $empty([1]);
                space.features_out(&(), &mut out, false);
                assert_eq!(out, $zeros([1]));
            }

            #[test]
            fn features_out_wrap_1() {
                let inner = IndexSpace::new(1);
                let space = NonEmptyFeatures::new(inner.clone());
                let mut out = $empty([1]);
                space.features_out(&0, &mut out, false);
                let expected: $array = inner.features(&0);
                assert_eq!(out, expected);
            }

            #[test]
            fn features_out_wrap_1_zeroed() {
                let inner = IndexSpace::new(1);
                let space = NonEmptyFeatures::new(inner.clone());
                let mut out: $array = $zeros([1]);
                space.features_out(&0, &mut out, true);
                let expected: $array = inner.features(&0);
                assert_eq!(out, expected);
            }

            #[test]
            fn features_out_wrap_2() {
                let inner = IndexSpace::new(2);
                let space = NonEmptyFeatures::new(inner.clone());
                let mut out = $empty([2]);
                space.features_out(&1, &mut out, false);
                let expected: $array = inner.features(&1);
                assert_eq!(out, expected);
            }
        };
    }

    mod tensor {
        use super::*;
        use tch::{Device, Kind, Tensor};

        tests!(
            Tensor,
            |s: [i64; 1]| Tensor::zeros(&s, (Kind::Float, Device::Cpu)),
            |s: [i64; 1]| Tensor::empty(&s, (Kind::Float, Device::Cpu))
        );
    }

    mod array {
        use super::*;
        use ndarray::{Array, Ix1};

        tests!(
            Array<f32, Ix1>,
            |s: [usize; 1]| Array::zeros(s),
            |s: [usize; 1]| Array::from_elem(s, f32::NAN) // Use NAN to detect any unset elements
        );
    }
}

#[cfg(test)]
mod batch_feature_space {
    use super::super::{IndexSpace, SingletonSpace};
    use super::*;

    macro_rules! tests {
        ($array:ty, $zeros:expr, $empty:expr) => {
            #[test]
            fn batch_features_wrap_0() {
                let space = NonEmptyFeatures::new(SingletonSpace::new());
                let actual: $array = space.batch_features(&[(), (), ()]);
                assert_eq!(actual, $zeros([3, 1]));
            }

            #[test]
            fn batch_features_wrap_1() {
                let inner = IndexSpace::new(1);
                let space = NonEmptyFeatures::new(inner.clone());
                let elements = [0, 0, 0];
                let actual: $array = space.batch_features(&elements);
                let expected: $array = inner.batch_features(&elements);
                assert_eq!(actual, expected);
            }

            #[test]
            fn batch_features_wrap_2() {
                let inner = IndexSpace::new(2);
                let space = NonEmptyFeatures::new(inner.clone());
                let elements = [1, 0, 1];
                let actual: $array = space.batch_features(&elements);
                let expected: $array = inner.batch_features(&elements);
                assert_eq!(actual, expected);
            }

            #[test]
            fn batch_features_out_wrap_0() {
                let space = NonEmptyFeatures::new(SingletonSpace::new());
                let mut out = $empty([3, 1]);
                space.batch_features_out(&[(), (), ()], &mut out, false);
                assert_eq!(out, $zeros([3, 1]));
            }

            #[test]
            fn batch_features_out_wrap_1() {
                let inner = IndexSpace::new(1);
                let space = NonEmptyFeatures::new(inner.clone());
                let mut out = $empty([3, 1]);
                let elements = [0, 0, 0];
                space.batch_features_out(&elements, &mut out, false);
                let expected: $array = inner.batch_features(&elements);
                assert_eq!(out, expected);
            }

            #[test]
            fn batch_features_out_wrap_1_zeroed() {
                let inner = IndexSpace::new(1);
                let space = NonEmptyFeatures::new(inner.clone());
                let mut out: $array = $zeros([3, 1]);
                let elements = [0, 0, 0];
                space.batch_features_out(&elements, &mut out, true);
                let expected: $array = inner.batch_features(&elements);
                assert_eq!(out, expected);
            }

            #[test]
            fn batch_features_out_wrap_2() {
                let inner = IndexSpace::new(2);
                let space = NonEmptyFeatures::new(inner.clone());
                let mut out = $empty([3, 2]);
                let elements = [1, 0, 1];
                space.batch_features_out(&elements, &mut out, false);
                let expected: $array = inner.batch_features(&elements);
                assert_eq!(out, expected);
            }
        };
    }

    mod tensor {
        use super::*;
        use tch::{Device, Kind, Tensor};

        tests!(
            Tensor,
            |s: [i64; 2]| Tensor::zeros(&s, (Kind::Float, Device::Cpu)),
            |s: [i64; 2]| Tensor::empty(&s, (Kind::Float, Device::Cpu))
        );
    }

    mod array {
        use super::*;
        use ndarray::{Array, Ix2};

        tests!(
            Array<f32, Ix2>,
            |s: [usize; 2]| Array::zeros(s),
            |s: [usize; 2]| Array::from_elem(s, f32::NAN) // Use NAN to detect any unset elements
        );
    }
}
