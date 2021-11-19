//! Generic wrapper spaces
use super::{
    ElementRefInto, EncoderFeatureSpace, FiniteSpace, NumFeatures, ReprSpace, Space, SubsetOrd,
};
use crate::utils::num_array::{BuildFromArray1D, BuildFromArray2D, NumArray1D, NumArray2D};
use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Float;
use rand::{distributions::Distribution, Rng};
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
        write!(f, "BoxSpace<{}>", self.inner_space)
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
    fn wrap(inner: Self::Inner) -> Self {
        inner.into()
    }
    fn inner_ref(&self) -> &Self::Inner {
        self
    }
}

/// Space that wraps the elements of an inner space without changing semantics.
pub struct WrappedElementSpace<S, W> {
    inner_space: S,
    wrapper: PhantomData<fn() -> W>,
}

impl<S, W> WrappedElementSpace<S, W> {
    pub fn new(inner_space: S) -> Self {
        Self {
            inner_space,
            wrapper: PhantomData,
        }
    }
}

impl<S, W> From<S> for WrappedElementSpace<S, W> {
    fn from(inner_space: S) -> Self {
        Self::new(inner_space)
    }
}

impl<S: fmt::Debug, W> fmt::Debug for WrappedElementSpace<S, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "WrappedElementSpace<{}>({:?})",
            any::type_name::<W>(),
            self.inner_space
        )
    }
}

impl<S: Default, W> Default for WrappedElementSpace<S, W> {
    fn default() -> Self {
        Self {
            inner_space: Default::default(),
            wrapper: PhantomData,
        }
    }
}

impl<S: Clone, W> Clone for WrappedElementSpace<S, W> {
    fn clone(&self) -> Self {
        Self {
            inner_space: self.inner_space.clone(),
            wrapper: PhantomData,
        }
    }
}

impl<S: Copy, W> Copy for WrappedElementSpace<S, W> {}

impl<S: PartialEq, W> PartialEq for WrappedElementSpace<S, W> {
    fn eq(&self, other: &Self) -> bool {
        self.inner_space.eq(&other.inner_space)
    }
}

impl<S: Eq, W> Eq for WrappedElementSpace<S, W> {}

impl<S: Hash, W> Hash for WrappedElementSpace<S, W> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner_space.hash(state)
    }
}

impl<S: PartialOrd, W> PartialOrd for WrappedElementSpace<S, W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner_space.partial_cmp(&other.inner_space)
    }
}

impl<S: Ord, W> Ord for WrappedElementSpace<S, W> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner_space.cmp(&other.inner_space)
    }
}

impl<S: SubsetOrd, W> SubsetOrd for WrappedElementSpace<S, W> {
    fn subset_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner_space.subset_cmp(&other.inner_space)
    }
}

impl<S, W> Space for WrappedElementSpace<S, W>
where
    S: Space,
    W: Wrapper<Inner = S::Element>,
{
    type Element = W;

    fn contains(&self, value: &Self::Element) -> bool {
        self.inner_space.contains(value.inner_ref())
    }
}

impl<S, W> FiniteSpace for WrappedElementSpace<S, W>
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

impl<S, W> Distribution<<Self as Space>::Element> for WrappedElementSpace<S, W>
where
    S: Space + Distribution<S::Element>,
    W: Wrapper<Inner = S::Element>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as Space>::Element {
        W::wrap(self.inner_space.sample(rng))
    }
}

impl<S, W, T, T0> ReprSpace<T, T0> for WrappedElementSpace<S, W>
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
        I::IntoIter: ExactSizeIterator + Clone,
        Self::Element: 'a,
    {
        self.inner_space
            .batch_repr(elements.into_iter().map(Wrapper::inner_ref))
    }
}

impl<S, W> NumFeatures for WrappedElementSpace<S, W>
where
    S: Space + NumFeatures,
    W: Wrapper<Inner = S::Element>,
{
    fn num_features(&self) -> usize {
        self.inner_space.num_features()
    }
}

impl<S, W> EncoderFeatureSpace for WrappedElementSpace<S, W>
where
    S: Space + EncoderFeatureSpace,
    S::Element: 'static,
    W: Wrapper<Inner = S::Element>,
{
    type Encoder = S::Encoder;

    fn encoder(&self) -> Self::Encoder {
        self.inner_space.encoder()
    }

    fn encoder_features_out<F: Float>(
        &self,
        element: &Self::Element,
        out: &mut [F],
        zeroed: bool,
        encoder: &Self::Encoder,
    ) {
        self.inner_space
            .encoder_features_out(element.inner_ref(), out, zeroed, encoder)
    }

    fn encoder_features<T>(&self, element: &Self::Element, encoder: &Self::Encoder) -> T
    where
        T: BuildFromArray1D,
        <T::Array as NumArray1D>::Elem: Float,
    {
        self.inner_space
            .encoder_features(element.inner_ref(), encoder)
    }

    fn encoder_batch_features_out<'a, I, A>(
        &self,
        elements: I,
        out: &mut ArrayBase<A, Ix2>,
        zeroed: bool,
        encoder: &Self::Encoder,
    ) where
        I: IntoIterator<Item = &'a Self::Element>,
        Self::Element: 'a,
        A: DataMut,
        A::Elem: Float,
    {
        self.inner_space.encoder_batch_features_out(
            elements.into_iter().map(Wrapper::inner_ref),
            out,
            zeroed,
            encoder,
        )
    }

    fn encoder_batch_features<'a, I, T>(&self, elements: I, encoder: &Self::Encoder) -> T
    where
        I: IntoIterator<Item = &'a Self::Element>,
        I::IntoIter: ExactSizeIterator,
        Self::Element: 'a,
        T: BuildFromArray2D,
        <T::Array as NumArray2D>::Elem: Float,
    {
        self.inner_space
            .encoder_batch_features(elements.into_iter().map(Wrapper::inner_ref), encoder)
    }
}

impl<S, W, T> ElementRefInto<T> for WrappedElementSpace<S, W>
where
    S: ElementRefInto<T>,
    W: Wrapper<Inner = S::Element>,
{
    fn elem_ref_into(&self, element: &Self::Element) -> T {
        self.inner_space.elem_ref_into(element.inner_ref())
    }
}
