//! History buffer implementation for an iterator of buffers.
use super::HistoryBuffer;
use crate::simulation::FullStep;
use std::iter::FusedIterator;

impl<T, O, A> HistoryBuffer<O, A> for Vec<T>
where
    T: HistoryBuffer<O, A>,
{
    fn num_steps(&self) -> usize {
        self.iter().map(HistoryBuffer::num_steps).sum()
    }

    fn num_episodes(&self) -> usize {
        self.iter().map(HistoryBuffer::num_episodes).sum()
    }

    fn steps<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a FullStep<O, A>> + 'a> {
        Box::new(SizedFlatMap::new(self.iter(), HistoryBuffer::steps))
    }

    fn drain_steps(&mut self) -> Box<dyn ExactSizeIterator<Item = FullStep<O, A>> + '_> {
        unimplemented!("hard to evaluate exact size")
    }

    fn episodes<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a [FullStep<O, A>]> + 'a> {
        Box::new(SizedFlatMap::new(self.iter(), HistoryBuffer::episodes))
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
    use crate::agents::FullStep;
    use crate::envs::Successor::{self, Continue, Terminate};
    use rstest::{fixture, rstest};

    /// Make a step that either continues or is terminal.
    const fn step(observation: usize, next: Successor<usize>) -> FullStep<usize, bool> {
        FullStep {
            observation,
            action: false,
            reward: 0.0,
            next,
        }
    }

    #[fixture]
    fn buffers() -> Vec<SerialBuffer<usize, bool>> {
        let config = SerialBufferConfig {
            soft_threshold: 4,
            hard_threshold: 4,
        };
        let mut b1 = config.build_history_buffer();
        b1.extend([
            step(0, Continue(1)),
            step(1, Continue(2)),
            step(2, Terminate),
            step(3, Continue(4)),
        ]);

        let mut b2 = config.build_history_buffer();
        b2.extend([
            step(10, Continue(11)),
            step(11, Terminate),
            step(12, Continue(13)),
            step(13, Continue(14)),
        ]);
        vec![b1, b2]
    }

    #[rstest]
    fn num_steps(buffers: Vec<SerialBuffer<usize, bool>>) {
        assert_eq!(buffers.num_steps(), 8);
    }

    #[rstest]
    fn num_episodes(buffers: Vec<SerialBuffer<usize, bool>>) {
        assert_eq!(buffers.num_episodes(), 4);
    }

    #[rstest]
    fn steps(buffers: Vec<SerialBuffer<usize, bool>>) {
        let mut steps_iter = buffers.steps();
        assert_eq!(steps_iter.next(), Some(&step(0, Continue(1))));
        assert_eq!(steps_iter.next(), Some(&step(1, Continue(2))));
        assert_eq!(steps_iter.next(), Some(&step(2, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(3, Continue(4))));
        assert_eq!(steps_iter.next(), Some(&step(10, Continue(11))));
        assert_eq!(steps_iter.next(), Some(&step(11, Terminate)));
        assert_eq!(steps_iter.next(), Some(&step(12, Continue(13))));
        assert_eq!(steps_iter.next(), Some(&step(13, Continue(14))));
        assert_eq!(steps_iter.next(), None);
    }

    #[rstest]
    fn steps_len(buffers: Vec<SerialBuffer<usize, bool>>) {
        assert_eq!(buffers.steps().len(), buffers.num_steps());
    }

    #[rstest]
    fn steps_is_fused(buffers: Vec<SerialBuffer<usize, bool>>) {
        let mut steps_iter = buffers.steps();
        for _ in 0..8 {
            assert!(steps_iter.next().is_some());
        }
        assert!(steps_iter.next().is_none());
        assert!(steps_iter.next().is_none());
    }

    #[rstest]
    fn episodes(buffers: Vec<SerialBuffer<usize, bool>>) {
        let mut episodes_iter = buffers.episodes();
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![
                &step(0, Continue(1)),
                &step(1, Continue(2)),
                &step(2, Terminate)
            ]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(3, Continue(4))]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(10, Continue(11)), &step(11, Terminate)]
        );
        assert_eq!(
            episodes_iter.next().unwrap().iter().collect::<Vec<_>>(),
            vec![&step(12, Continue(13)), &step(13, Continue(14))]
        );
        assert!(episodes_iter.next().is_none());
    }

    #[rstest]
    fn episodes_is_fused(buffers: Vec<SerialBuffer<usize, bool>>) {
        let mut episodes_iter = buffers.episodes();
        for _ in 0..4 {
            assert!(episodes_iter.next().is_some());
        }
        assert!(episodes_iter.next().is_none());
        assert!(episodes_iter.next().is_none());
    }

    #[rstest]
    fn episodes_len(buffers: Vec<SerialBuffer<usize, bool>>) {
        assert_eq!(buffers.episodes().len(), buffers.num_episodes());
    }

    #[rstest]
    fn episode_len_sum(buffers: Vec<SerialBuffer<usize, bool>>) {
        assert_eq!(
            buffers.episodes().map(|e| e.len()).sum::<usize>(),
            buffers.num_steps()
        );
    }
}
