//! Generic wrapper spaces
use super::{
    FeatureSpace, FiniteSpace, LogElementSpace, NonEmptySpace, ReprSpace, Space, SubsetOrd,
};
use crate::logging::{LogError, StatsLogger};
use crate::utils::num_array::{BuildFromArray1D, BuildFromArray2D, NumArray1D, NumArray2D};
use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Float;
use rand::{distributions::Distribution, Rng};
use serde::{Deserialize, Serialize};
use std::any;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;

/// A wrapper space that boxes the elements of an inner space.
pub type BoxSpace<S> = WrappedElementSpace<S, Box<<S as Space>::Element>>;

impl<S: Space + fmt::Display> fmt::Display for BoxSpace<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BoxSpace<{}>", self.inner)
    }
}

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
    #[inline]
    fn wrap(inner: Self::Inner) -> Self {
        inner.into()
    }
    #[inline]
    fn inner_ref(&self) -> &Self::Inner {
        self
    }
}

/// Space that wraps the elements of an inner space without changing semantics.
// Can only use derives for serde traits
// because they allow modifying the bounds to remove `W: <Trait>`
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: Deserialize<'de>"))]
pub struct WrappedElementSpace<S, W> {
    inner: S,
    #[serde(skip)]
    wrapper: PhantomData<fn() -> W>,
}

impl<S, W> WrappedElementSpace<S, W> {
    #[inline]
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            wrapper: PhantomData,
        }
    }
}

impl<S, W> From<S> for WrappedElementSpace<S, W> {
    #[inline]
    fn from(inner: S) -> Self {
        Self::new(inner)
    }
}

impl<S: fmt::Debug, W> fmt::Debug for WrappedElementSpace<S, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "WrappedElementSpace<{}>({:?})",
            any::type_name::<W>(),
            self.inner
        )
    }
}

impl<S: Default, W> Default for WrappedElementSpace<S, W> {
    #[inline]
    fn default() -> Self {
        Self {
            inner: Default::default(),
            wrapper: PhantomData,
        }
    }
}

impl<S: Clone, W> Clone for WrappedElementSpace<S, W> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            wrapper: PhantomData,
        }
    }
}

impl<S: Copy, W> Copy for WrappedElementSpace<S, W> {}

impl<S: PartialEq, W> PartialEq for WrappedElementSpace<S, W> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl<S: Eq, W> Eq for WrappedElementSpace<S, W> {}

impl<S: Hash, W> Hash for WrappedElementSpace<S, W> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl<S: PartialOrd, W> PartialOrd for WrappedElementSpace<S, W> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl<S: Ord, W> Ord for WrappedElementSpace<S, W> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl<S: SubsetOrd, W> SubsetOrd for WrappedElementSpace<S, W> {
    #[inline]
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.subset_cmp(&other.inner)
    }
}

impl<S, W> Space for WrappedElementSpace<S, W>
where
    S: Space,
    W: Wrapper<Inner = S::Element> + Clone + Send,
{
    type Element = W;

    #[inline]
    fn contains(&self, value: &Self::Element) -> bool {
        self.inner.contains(value.inner_ref())
    }
}

impl<S, W> FiniteSpace for WrappedElementSpace<S, W>
where
    S: FiniteSpace,
    W: Wrapper<Inner = S::Element> + Clone + Send,
{
    #[inline]
    fn size(&self) -> usize {
        self.inner.size()
    }

    #[inline]
    fn to_index(&self, element: &Self::Element) -> usize {
        self.inner.to_index(element.inner_ref())
    }

    #[inline]
    fn from_index(&self, index: usize) -> Option<Self::Element> {
        self.inner.from_index(index).map(Wrapper::wrap)
    }

    #[inline]
    fn from_index_unchecked(&self, index: usize) -> Option<Self::Element> {
        self.inner.from_index_unchecked(index).map(Wrapper::wrap)
    }
}

impl<S, W> NonEmptySpace for WrappedElementSpace<S, W>
where
    S: NonEmptySpace,
    W: Wrapper<Inner = S::Element> + Clone + Send,
{
    #[inline]
    fn some_element(&self) -> Self::Element {
        W::wrap(self.inner.some_element())
    }
}

impl<S, W> Distribution<<Self as Space>::Element> for WrappedElementSpace<S, W>
where
    S: Space + Distribution<S::Element>,
    W: Wrapper<Inner = S::Element> + Clone + Send,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        W::wrap(self.inner.sample(rng))
    }
}

impl<S, W, T, T0> ReprSpace<T, T0> for WrappedElementSpace<S, W>
where
    S: ReprSpace<T, T0>,
    S::Element: 'static,
    W: Wrapper<Inner = S::Element> + Clone + Send,
{
    #[inline]
    fn repr(&self, element: &Self::Element) -> T0 {
        self.inner.repr(element.inner_ref())
    }

    #[inline]
    fn batch_repr<'a, I>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        self.inner
            .batch_repr(elements.into_iter().map(Wrapper::inner_ref))
    }
}

impl<S, W> FeatureSpace for WrappedElementSpace<S, W>
where
    S: FeatureSpace,
    S::Element: 'static,
    W: Wrapper<Inner = S::Element> + Clone + Send,
{
    #[inline]
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    #[inline]
    fn features_out<'a, F: Float>(
        &self,
        element: &Self::Element,
        out: &'a mut [F],
        zeroed: bool,
    ) -> &'a mut [F] {
        self.inner.features_out(element.inner_ref(), out, zeroed)
    }
    #[inline]
    fn features<T>(&self, element: &Self::Element) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        self.inner.features(element.inner_ref())
    }
    #[inline]
    fn batch_features_out<'a, I, A>(&self, elements: I, out: &mut ArrayBase<A, Ix2>, zeroed: bool)
    where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
        self.inner
            .batch_features_out(elements.into_iter().map(Wrapper::inner_ref), out, zeroed)
    }
    #[inline]
    fn batch_features<'a, I, T>(&self, elements: I) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
        T: BuildFromArray2D,
        <T::Array as NumArray2D>::Elem: Float,
    {
        self.inner
            .batch_features(elements.into_iter().map(Wrapper::inner_ref))
    }
}

impl<S, W> LogElementSpace for WrappedElementSpace<S, W>
where
    S: LogElementSpace,
    W: Wrapper<Inner = S::Element> + Clone + Send,
{
    #[inline]
    fn log_element<L: StatsLogger + ?Sized>(
        &self,
        name: &'static str,
        element: &Self::Element,
        logger: &mut L,
    ) -> Result<(), LogError> {
        self.inner.log_element(name, element.inner_ref(), logger)
    }
}
