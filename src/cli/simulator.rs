use super::{Options, Update, WithUpdate};
use crate::simulation::{MultiThreadSimulatorConfig, SimulatorConfig};

impl From<&Options> for SimulatorConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for SimulatorConfig {
    fn update(&mut self, opts: &Options) {
        self.seed = opts.seed;
    }
}

impl From<&Options> for MultiThreadSimulatorConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for MultiThreadSimulatorConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(num_workers) = opts.parallel_threads {
            if num_workers == 0 {
                self.num_workers = num_cpus::get();
            } else {
                self.num_workers = num_workers;
            }
        }
    }
}
