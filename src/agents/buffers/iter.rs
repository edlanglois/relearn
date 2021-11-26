//! History buffer implementation for an iterator of buffers.
use super::{
    EpisodesIter, HistoryBufferBoxedEpisodes, HistoryBufferBoxedSteps, HistoryBufferEpisodes,
    HistoryBufferSteps, StepsIter,
};
use std::iter::FusedIterator;

pub type StepsIter_<'a, O, A, I, T> = SizedFlatMap<
    <&'a I as IntoIterator>::IntoIter,
    <T as HistoryBufferSteps<'a, O, A>>::StepsIter,
    fn(&'a T) -> <T as HistoryBufferSteps<'a, O, A>>::StepsIter,
>;

impl<'a, O: 'a, A: 'a, I: 'a, T: 'a> HistoryBufferSteps<'a, O, A> for I
where
    &'a I: IntoIterator<Item = &'a T>,
    <&'a I as IntoIterator>::IntoIter: Clone,
    T: HistoryBufferSteps<'a, O, A>,
{
    type StepsIter = StepsIter_<'a, O, A, I, T>;

    fn steps_(&'a self) -> Self::StepsIter {
        SizedFlatMap::new(self.into_iter(), HistoryBufferSteps::steps_)
    }
}

// Would be implemented with a generic impl on HistoryBufferSteps
// except that requires O: 'static and A: 'static
impl<O, A, I, T> HistoryBufferBoxedSteps<O, A> for I
where
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    for<'a> <&'a I as IntoIterator>::IntoIter: Clone,
    T: HistoryBufferBoxedSteps<O, A>,
{
    fn steps<'a>(&'a self) -> Box<dyn StepsIter<O, A> + 'a>
    where
        O: 'a,
        A: 'a,
    {
        Box::new(SizedFlatMap::new(self.into_iter(), |b: &'a T| b.steps()))
    }
}

pub type EpisodesIter_<'a, O, A, I, T> = SizedFlatMap<
    <&'a I as IntoIterator>::IntoIter,
    <T as HistoryBufferEpisodes<'a, O, A>>::EpisodesIter,
    fn(&'a T) -> <T as HistoryBufferEpisodes<'a, O, A>>::EpisodesIter,
>;

impl<'a, O: 'a, A: 'a, I: 'a, T: 'a> HistoryBufferEpisodes<'a, O, A> for I
where
    &'a I: IntoIterator<Item = &'a T>,
    <&'a I as IntoIterator>::IntoIter: Clone,
    T: HistoryBufferEpisodes<'a, O, A>,
{
    type EpisodesIter = EpisodesIter_<'a, O, A, I, T>;

    fn episodes_(&'a self) -> Self::EpisodesIter {
        SizedFlatMap::new(self.into_iter(), HistoryBufferEpisodes::episodes_)
    }
}

impl<O, A, I, T> HistoryBufferBoxedEpisodes<O, A> for I
where
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    for<'a> <&'a I as IntoIterator>::IntoIter: Clone,
    T: HistoryBufferBoxedEpisodes<O, A>,
{
    fn episodes<'a>(&'a self) -> Box<dyn EpisodesIter<O, A> + 'a>
    where
        O: 'a,
        A: 'a,
    {
        Box::new(SizedFlatMap::new(self.into_iter(), |b: &'a T| b.episodes()))
    }
}

/// Like [`std::iter::FlatMap`] but with a clonable outer iterator.
///
/// This allows tighter size bounds by iterating over a copy of the outer iterator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SizedFlatMap<I, U: IntoIterator, F> {
    outer: I,
    inner: Option<U::IntoIter>,
    f: F,
}

impl<I, U: IntoIterator, F> SizedFlatMap<I, U, F> {
    pub fn new<T: IntoIterator<IntoIter = I>>(iter: T, f: F) -> Self {
        Self {
            outer: iter.into_iter(),
            inner: None,
            f,
        }
    }
}

impl<I, U, F> Iterator for SizedFlatMap<I, U, F>
where
    I: Iterator + Clone,
    U: IntoIterator,
    F: Fn(I::Item) -> U,
{
    type Item = <U as IntoIterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.inner.is_none() {
            self.inner = Some((self.f)(self.outer.next()?).into_iter());
        }
        let inner = self.inner.as_mut().unwrap();
        loop {
            match inner.next() {
                Some(x) => return Some(x),
                None => *inner = (self.f)(self.outer.next()?).into_iter(),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let inner_size_hint = self
            .inner
            .as_ref()
            .map_or((0, Some(0)), Iterator::size_hint);
        self.outer.clone().fold(inner_size_hint, |(lb, ub), item| {
            let (item_lb, item_ub) = (self.f)(item).into_iter().size_hint();
            let new_lb = lb + item_lb;
            let new_ub = ub.zip(item_ub).map(|(a, b)| a + b);
            (new_lb, new_ub)
        })
    }

    fn count(self) -> usize {
        self.outer
            .fold(self.inner.map_or(0, Iterator::count), |accum, item| {
                accum + (self.f)(item).into_iter().count()
            })
    }

    fn fold<B, G>(self, init: B, mut g: G) -> B
    where
        G: FnMut(B, Self::Item) -> B,
    {
        let inner_result = match self.inner {
            Some(inner) => inner.fold(init, &mut g),
            None => init,
        };
        self.outer.fold(inner_result, |accum, item| {
            (self.f)(item).into_iter().fold(accum, &mut g)
        })
    }
}

impl<I, U, F> FusedIterator for SizedFlatMap<I, U, F>
where
    I: Iterator + FusedIterator + Clone,
    U: IntoIterator,
    F: Fn(I::Item) -> U,
{
}

impl<I, U, F> ExactSizeIterator for SizedFlatMap<I, U, F>
where
    I: Iterator + Clone,
    U: IntoIterator,
    U::IntoIter: ExactSizeIterator,
    F: Fn(I::Item) -> U,
{
    fn len(&self) -> usize {
        self.outer.clone().fold(
            self.inner.as_ref().map_or(0, ExactSizeIterator::len),
            |accum, item| accum + (self.f)(item).into_iter().len(),
        )
    }
}

#[allow(clippy::needless_pass_by_value)]
#[cfg(test)]
mod tests {
    use super::super::{BuildHistoryBuffer, SerialBuffer, SerialBufferConfig};
    use super::*;
    use crate::agents::Step;
    use rstest::{fixture, rstest};

    /// Make a step that either continues or is terminal.
    const fn step(observation: usize, next_observation: Option<usize>) -> Step<usize, bool> {
        Step {
            observation,
            action: false,
            reward: 0.0,
            next_observation,
            episode_done: next_observation.is_none(),
        }
    }

    #[fixture]
    fn buffers() -> [SerialBuffer<usize, bool>; 2] {
        let config = SerialBufferConfig {
            soft_threshold: 4,
            hard_threshold: 4,
        };
        let mut b1 = config.build_history_buffer();
        b1.extend([
            step(0, Some(1)),
            step(1, Some(2)),
            step(2, None),
            step(3, Some(4)),
        ]);

        let mut b2 = config.build_history_buffer();
        b2.extend([
            step(10, Some(11)),
            step(11, None),
            step(12, Some(13)),
            step(13, Some(14)),
        ]);
        [b1, b2]
    }

    #[rstest]
    fn steps(buffers: [SerialBuffer<usize, bool>; 2]) {
        let mut steps_iter = buffers.steps_();
        assert_eq!(steps_iter.next(), Some(&step(0, Some(1))));
        assert_eq!(steps_iter.next(), Some(&step(1, Some(2))));
        assert_eq!(steps_iter.next(), Some(&step(2, None)));
        assert_eq!(steps_iter.next(), Some(&step(3, Some(4))));
        assert_eq!(steps_iter.next(), Some(&step(10, Some(11))));
        assert_eq!(steps_iter.next(), Some(&step(11, None)));
        assert_eq!(steps_iter.next(), Some(&step(12, Some(13))));
        assert_eq!(steps_iter.next(), Some(&step(13, Some(14))));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_len(buffers: [SerialBuffer<usize, bool>; 2]) {
        assert_eq!(buffers.steps_().len(), 8);
    }

    #[rstest]
    fn steps_is_fused(buffers: [SerialBuffer<usize, bool>; 2]) {
        let mut steps_iter = buffers.steps_();
        for _ in 0..8 {
            assert!(steps_iter.next().is_some());
        }
        assert!(steps_iter.next().is_none());
        assert!(steps_iter.next().is_none());
    }

    #[rstest]
    fn episodes_all_incomplete(buffers: [SerialBuffer<usize, bool>; 2]) {
        let mut episodes_iter = buffers.episodes_();
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(0, Some(1)), &step(1, Some(2)), &step(2, None)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(3, Some(4))]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(10, Some(11)), &step(11, None)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(12, Some(13)), &step(13, Some(14))]
        );
        assert!(episodes_iter.next().is_none());
    }

    #[rstest]
    fn episodes_is_fused(buffers: [SerialBuffer<usize, bool>; 2]) {
        let mut episodes_iter = buffers.episodes_();
        for _ in 0..4 {
            assert!(episodes_iter.next().is_some());
        }
        assert!(episodes_iter.next().is_none());
        assert!(episodes_iter.next().is_none());
    }
}
