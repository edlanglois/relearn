use super::{Options, Update, WithUpdate};
use crate::logging::CLILoggerConfig;
use std::time::Duration;

impl From<&Options> for CLILoggerConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for CLILoggerConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(display_period) = opts.display_period {
            self.display_period = Duration::from_secs(display_period);
            self.urgent_display_period = self.display_period.mul_f64(1.1);
        }
    }
}
