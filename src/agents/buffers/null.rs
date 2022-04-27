use super::{WriteExperience, WriteExperienceError, WriteExperienceIncremental};
use crate::simulation::PartialStep;

/// Buffer that drops all steps without saving.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NullBuffer;

impl<O, A> WriteExperience<O, A> for NullBuffer {
    fn write_experience<I>(&mut self, _: I) -> Result<(), WriteExperienceError>
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
        Ok(())
    }
}

impl<O, A> WriteExperienceIncremental<O, A> for NullBuffer {
    fn write_step(&mut self, _: PartialStep<O, A>) -> Result<(), WriteExperienceError> {
        Ok(())
    }
    fn end_experience(&mut self) {}
}
