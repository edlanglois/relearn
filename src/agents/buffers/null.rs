use super::WriteHistoryBuffer;
use crate::simulation::PartialStep;

/// Buffer that drops all steps without saving. Always reports being ready.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NullBuffer;

impl<O, A> WriteHistoryBuffer<O, A> for NullBuffer {
    fn push(&mut self, _: PartialStep<O, A>) {}
    fn extend<I: IntoIterator<Item = PartialStep<O, A>>>(&mut self, _: I) {}
}
