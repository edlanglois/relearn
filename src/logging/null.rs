use super::{Event, LogError, Loggable, Logger};

/// A logger that does nothing.
pub struct NullLogger;

impl NullLogger {
    pub fn new() -> Self {
        Self {}
    }
}

impl Logger for NullLogger {
    fn log(
        &mut self,
        _event: Event,
        _name: &'static str,
        _value: Loggable,
    ) -> Result<(), LogError> {
        Ok(())
    }

    fn done(&mut self, _event: Event) {}
}
