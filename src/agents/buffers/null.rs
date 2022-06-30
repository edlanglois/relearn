use super::{WriteExperience, WriteExperienceError, WriteExperienceIncremental};
use crate::simulation::PartialStep;

/// Buffer that drops all steps without saving.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NullBuffer;

impl<O, A, F> WriteExperience<O, A, F> for NullBuffer {
    fn write_experience<I>(&mut self, _: I) -> Result<(), WriteExperienceError>
    where
        I: IntoIterator<Item = PartialStep<O, A, F>>,
    {
        Ok(())
    }
}

impl<O, A, F> WriteExperienceIncremental<O, A, F> for NullBuffer {
    fn write_step(&mut self, _: PartialStep<O, A, F>) -> Result<(), WriteExperienceError> {
        Ok(())
    }
    fn end_experience(&mut self) {}
}
