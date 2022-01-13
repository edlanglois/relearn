use std::iter::{Chain, FusedIterator};

/// [`Chain`] wrapper that implements [`ExactSizeIterator`].
///
/// See [here](https://github.com/rust-lang/rust/issues/34433) for a discussion of why the builtin
/// `Chain` does not implement `ExactSizeIterator`.
/// This implementation panics in `ExactSizeIterator::len` if the combined length is larger than
/// `usize::MAX`. `SizedChain::size_hint` will return `None` as an upper bound in that case, which
/// is technically a violation of the conditions of `ExactSizeIterator` but returning `None` when
/// expecting `Some` is unlikely to cause any trouble other than a panic.
#[derive(Debug, Clone)]
pub struct SizedChain<A, B> {
    chain: Chain<A, B>,
}

impl<A, B> From<Chain<A, B>> for SizedChain<A, B> {
    fn from(chain: Chain<A, B>) -> Self {
        Self { chain }
    }
}

impl<A, B> SizedChain<A, B>
where
    A: Iterator,
{
    pub fn new<I>(a: A, b: I) -> Self
    where
        I: IntoIterator<IntoIter = B, Item = A::Item>,
    {
        Self { chain: a.chain(b) }
    }
}

impl<A, B> Iterator for SizedChain<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.chain.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chain.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.chain.count()
    }

    #[inline]
    fn fold<Acc, F>(self, acc: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.chain.fold(acc, f)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.chain.nth(n)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.chain.find(predicate)
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.chain.last()
    }
}

// The default implementation is already fine. It will panic if the returned upper bound is None.
impl<A, B> ExactSizeIterator for SizedChain<A, B>
where
    A: ExactSizeIterator,
    B: ExactSizeIterator<Item = A::Item>,
{
}

impl<A, B> DoubleEndedIterator for SizedChain<A, B>
where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item = A::Item>,
{
    #[inline]
    fn next_back(&mut self) -> Option<A::Item> {
        self.chain.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.chain.nth_back(n)
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.chain.rfind(predicate)
    }

    fn rfold<Acc, F>(self, acc: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.chain.rfold(acc, f)
    }
}

impl<A, B> FusedIterator for SizedChain<A, B>
where
    A: FusedIterator,
    B: FusedIterator<Item = A::Item>,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len() {
        let iter: SizedChain<_, _> = (0..3).chain(10..15).into();
        assert_eq!(iter.len(), 8);
    }

    #[test]
    #[should_panic]
    fn len_overflow() {
        let iter: SizedChain<_, _> = (0..usize::MAX).chain(0..usize::MAX).into();
        iter.len();
    }
}
