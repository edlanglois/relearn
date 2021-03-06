use super::WriteHistoryBuffer;
use crate::simulation::PartialStep;

/// Buffer that drops all steps without saving. Always reports being ready.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NullBuffer;

impl<O, A> WriteHistoryBuffer<O, A> for NullBuffer {
    fn push(&mut self, _step: PartialStep<O, A>) -> bool {
        true
    }

    fn extend_until_ready<I>(&mut self, _steps: I) -> bool
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
        true
    }

    fn clear(&mut self) {}
}
