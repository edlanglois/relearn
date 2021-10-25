//! Generic wrapper spaces
use super::{
    BaseFeatureSpace, BatchFeatureSpace, BatchFeatureSpaceOut, ElementRefInto, FeatureSpace,
    FeatureSpaceOut, FiniteSpace, ReprSpace, Space,
};
use rand::{distributions::Distribution, Rng};
use std::marker::PhantomData;
use std::ops::Deref;

/// A wrapper space that boxes the elements of an inner space.
pub type BoxSpace<S> = WrapperSpace<S, Box<<S as Space>::Element>>;

/// Wraps an inner type
pub trait Wrapper {
    type Inner;

    /// Wrap an object
    fn wrap(inner: Self::Inner) -> Self;

    /// Return a reference to the inner object
    fn inner_ref(&self) -> &Self::Inner;
}

/// Auto-implement wrapper for smart pointers
impl<T> Wrapper for T
where
    T: Deref + From<T::Target>,
    T::Target: Sized,
{
    type Inner = T::Target;
    fn wrap(inner: Self::Inner) -> Self {
        inner.into()
    }
    fn inner_ref(&self) -> &Self::Inner {
        self
    }
}

/// Wrapper space that wraps the elements of an inner space without changing semantics.
pub struct WrapperSpace<S, W> {
    inner_space: S,
    wrapper: PhantomData<*const W>,
}

impl<S, W> Space for WrapperSpace<S, W>
where
    S: Space,
    W: Wrapper<Inner = S::Element>,
{
    type Element = W;

    fn contains(&self, value: &Self::Element) -> bool {
        self.inner_space.contains(value.inner_ref())
    }
}

impl<S, W> FiniteSpace for WrapperSpace<S, W>
where
    S: FiniteSpace,
    W: Wrapper<Inner = S::Element>,
{
    fn size(&self) -> usize {
        self.inner_space.size()
    }

    fn to_index(&self, element: &Self::Element) -> usize {
        self.inner_space.to_index(element.inner_ref())
    }

    fn from_index(&self, index: usize) -> Option<Self::Element> {
        self.inner_space.from_index(index).map(Wrapper::wrap)
    }

    fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
        self.inner_space
            .from_index_unchecked(index)
            .map(Wrapper::wrap)
    }
}

impl<S, W> Distribution<<Self as Space>::Element> for WrapperSpace<S, W>
where
    S: Space + Distribution<S::Element>,
    W: Wrapper<Inner = S::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        W::wrap(self.inner_space.sample(rng))
    }
}

impl<S, W, T, T0> ReprSpace<T, T0> for WrapperSpace<S, W>
where
    S: ReprSpace<T, T0>,
    S::Element: 'static,
    W: Wrapper<Inner = S::Element>,
{
    fn repr(&self, element: &Self::Element) -> T0 {
        self.inner_space.repr(element.inner_ref())
    }

    fn batch_repr<'a, I>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
    {
        self.inner_space
            .batch_repr(elements.into_iter().map(Wrapper::inner_ref))
    }
}

impl<S, W> BaseFeatureSpace for WrapperSpace<S, W>
where
    S: Space + BaseFeatureSpace,
    W: Wrapper<Inner = S::Element>,
{
    fn num_features(&self) -> usize {
        self.inner_space.num_features()
    }
}

impl<S, W, T> FeatureSpace<T> for WrapperSpace<S, W>
where
    S: FeatureSpace<T>,
    W: Wrapper<Inner = S::Element>,
{
    fn features(&self, element: &Self::Element) -> T {
        self.inner_space.features(element.inner_ref())
    }
}

impl<S, W, T> FeatureSpaceOut<T> for WrapperSpace<S, W>
where
    S: FeatureSpaceOut<T>,
    W: Wrapper<Inner = S::Element>,
{
    fn features_out(&self, element: &Self::Element, out: &mut T, zeroed: bool) {
        self.inner_space
            .features_out(element.inner_ref(), out, zeroed);
    }
}

impl<S, W, T2> BatchFeatureSpace<T2> for WrapperSpace<S, W>
where
    S: BatchFeatureSpace<T2>,
    S::Element: 'static,
    W: Wrapper<Inner = S::Element>,
{
    fn batch_features<'a, I>(&self, elements: I) -> T2
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        self.inner_space
            .batch_features(elements.into_iter().map(Wrapper::inner_ref))
    }
}

impl<S, W, T2> BatchFeatureSpaceOut<T2> for WrapperSpace<S, W>
where
    S: BatchFeatureSpaceOut<T2>,
    S::Element: 'static,
    W: Wrapper<Inner = S::Element>,
{
    fn batch_features_out<'a, I>(&self, elements: I, out: &mut T2, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: Clone,
        Self::Element: 'a,
    {
        self.inner_space.batch_features_out(
            elements.into_iter().map(Wrapper::inner_ref),
            out,
            zeroed,
        );
    }
}

impl<S, W, T> ElementRefInto<T> for WrapperSpace<S, W>
where
    S: ElementRefInto<T>,
    W: Wrapper<Inner = S::Element>,
{
    fn elem_ref_into(&self, element: &Self::Element) -> T {
        self.inner_space.elem_ref_into(element.inner_ref())
    }
}
