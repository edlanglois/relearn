//! Wrap spaces to have non-empty feature vectors.
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, FeatureSpace, FeatureSpaceOut,
    FiniteSpace, Space,
};
use std::fmt;
use tch::{Device, Kind, Tensor};

// Note: This could be renamed to something like TorchSpace if other helper changes end up being
// requried to satisfy torch methods.

/// Wrapper space with a feature vector length of at least 1.
///
/// Any generated features have value `0`.
#[derive(Debug, Clone)]
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

impl<S: FeatureSpace<Tensor>> FeatureSpace<Tensor> for NonEmptyFeatures<S> {
    fn features(&self, element: &Self::Element) -> Tensor {
        if self.inner.num_features() == 0 {
            Tensor::zeros(&[1], (Kind::Float, Device::Cpu))
        } else {
            self.inner.features(element)
        }
    }
}

impl<S: BatchFeatureSpace<Tensor>> BatchFeatureSpace<Tensor> for NonEmptyFeatures<S> {
    fn batch_features<'a, I>(&self, elements: I) -> Tensor
    where
        I: IntoIterator<Item = &'a Self::Element>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
    {
        if self.inner.num_features() == 0 {
            let num_elements = elements.into_iter().count();
            Tensor::zeros(&[num_elements as i64, 1], (Kind::Float, Device::Cpu))
        } else {
            self.inner.batch_features(elements)
        }
    }
}

impl<S: FeatureSpaceOut<Tensor>> FeatureSpaceOut<Tensor> for NonEmptyFeatures<S> {
    fn features_out(&self, element: &Self::Element, out: &mut Tensor, zeroed: bool) {
        if self.inner.num_features() == 0 {
            if !zeroed {
                let _ = out.zero_();
            }
        } else {
            self.inner.features_out(element, out, zeroed);
        }
    }
}

impl<S: BatchFeatureSpaceOut<Tensor>> BatchFeatureSpaceOut<Tensor> for NonEmptyFeatures<S> {
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut Tensor, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        if self.inner.num_features() == 0 {
            if !zeroed {
                let _ = out.zero_();
            }
        } else {
            self.inner.batch_features_out(elements, out, zeroed);
        }
    }
}

#[cfg(test)]
mod feature_space_tensor {
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

    #[test]
    fn features_wrap_0() {
        let space = NonEmptyFeatures::new(SingletonSpace::new());
        assert_eq!(
            space.features(&()),
            Tensor::zeros(&[1], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn features_wrap_1() {
        let inner = IndexSpace::new(1);
        let space = NonEmptyFeatures::new(inner.clone());
        assert_eq!(space.features(&0), inner.features(&0),);
    }

    #[test]
    fn features_wrap_2() {
        let inner = IndexSpace::new(2);
        let space = NonEmptyFeatures::new(inner.clone());
        assert_eq!(space.features(&1), inner.features(&1));
    }

    #[test]
    fn features_out_wrap_0() {
        let space = NonEmptyFeatures::new(SingletonSpace::new());
        let mut out = Tensor::empty(&[1], (Kind::Float, Device::Cpu));
        space.features_out(&(), &mut out, false);
        assert_eq!(out, Tensor::zeros(&[1], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn features_out_wrap_1() {
        let inner = IndexSpace::new(1);
        let space = NonEmptyFeatures::new(inner.clone());
        let mut out = Tensor::empty(&[1], (Kind::Float, Device::Cpu));
        space.features_out(&0, &mut out, false);
        assert_eq!(out, inner.features(&0),);
    }

    #[test]
    fn features_out_wrap_1_zeroed() {
        let inner = IndexSpace::new(1);
        let space = NonEmptyFeatures::new(inner.clone());
        let mut out = Tensor::zeros(&[1], (Kind::Float, Device::Cpu));
        space.features_out(&0, &mut out, true);
        assert_eq!(out, inner.features(&0),);
    }

    #[test]
    fn features_out_wrap_2() {
        let inner = IndexSpace::new(2);
        let space = NonEmptyFeatures::new(inner.clone());
        let mut out = Tensor::empty(&[2], (Kind::Float, Device::Cpu));
        space.features_out(&1, &mut out, false);
        assert_eq!(out, inner.features(&1));
    }

    #[test]
    fn batch_features_wrap_0() {
        let space = NonEmptyFeatures::new(SingletonSpace::new());
        assert_eq!(
            space.batch_features(&[(), (), ()]),
            Tensor::zeros(&[3, 1], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn batch_features_wrap_1() {
        let inner = IndexSpace::new(1);
        let space = NonEmptyFeatures::new(inner.clone());
        let elements = [0, 0, 0];
        assert_eq!(
            space.batch_features(&elements),
            inner.batch_features(&elements),
        );
    }

    #[test]
    fn batch_features_wrap_2() {
        let inner = IndexSpace::new(2);
        let space = NonEmptyFeatures::new(inner.clone());
        let elements = [1, 0, 1];
        assert_eq!(
            space.batch_features(&elements),
            inner.batch_features(&elements),
        );
    }

    #[test]
    fn batch_features_out_wrap_0() {
        let space = NonEmptyFeatures::new(SingletonSpace::new());
        let mut out = Tensor::empty(&[3, 1], (Kind::Float, Device::Cpu));
        space.batch_features_out(&[(), (), ()], &mut out, false);
        assert_eq!(out, Tensor::zeros(&[3, 1], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn batch_features_out_wrap_1() {
        let inner = IndexSpace::new(1);
        let space = NonEmptyFeatures::new(inner.clone());
        let mut out = Tensor::empty(&[3, 1], (Kind::Float, Device::Cpu));
        let elements = [0, 0, 0];
        space.batch_features_out(&elements, &mut out, false);
        assert_eq!(out, inner.batch_features(&elements),);
    }

    #[test]
    fn batch_features_out_wrap_1_zeroed() {
        let inner = IndexSpace::new(1);
        let space = NonEmptyFeatures::new(inner.clone());
        let mut out = Tensor::empty(&[3, 1], (Kind::Float, Device::Cpu));
        let elements = [0, 0, 0];
        space.batch_features_out(&elements, &mut out, true);
        assert_eq!(out, inner.batch_features(&elements),);
    }

    #[test]
    fn batch_features_out_wrap_2() {
        let inner = IndexSpace::new(2);
        let space = NonEmptyFeatures::new(inner.clone());
        let mut out = Tensor::empty(&[3, 2], (Kind::Float, Device::Cpu));
        let elements = [1, 0, 1];
        space.batch_features_out(&elements, &mut out, false);
        assert_eq!(out, inner.batch_features(&elements),);
    }
}
