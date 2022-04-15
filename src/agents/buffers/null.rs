use super::{WriteExperience, WriteExperienceIncremental};
use crate::simulation::PartialStep;

/// Buffer that drops all steps without saving. Always reports being ready.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NullBuffer;

impl<O, A> WriteExperience<O, A> for NullBuffer {
    fn write_experience<I>(&mut self, _: I)
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
    }
}

impl<O, A> WriteExperienceIncremental<O, A> for NullBuffer {
    fn write_step(&mut self, _: PartialStep<O, A>) {}
    fn end_experience(&mut self) {}
}
