use super::{Options, Update, WithUpdate};
use crate::defs::OptimizerDef;
use crate::torch::optimizers::{AdamConfig, AdamWConfig, RmsPropConfig, SgdConfig};
use clap::Clap;

/// Optimizer name
#[derive(Clap, Debug, PartialEq, Eq)]
pub enum OptimizerType {
    Sgd,
    RmsProp,
    Adam,
    AdamW,
}

impl OptimizerDef {
    pub fn type_(&self) -> OptimizerType {
        use OptimizerDef::*;
        match self {
            Sgd(_) => OptimizerType::Sgd,
            RmsProp(_) => OptimizerType::RmsProp,
            Adam(_) => OptimizerType::Adam,
            AdamW(_) => OptimizerType::AdamW,
        }
    }
}

impl From<&Options> for OptimizerDef {
    fn from(opts: &Options) -> Self {
        use OptimizerType::*;
        match opts.optimizer {
            Some(Sgd) => OptimizerDef::Sgd(opts.into()),
            Some(RmsProp) => OptimizerDef::RmsProp(opts.into()),
            Some(Adam) | None => OptimizerDef::Adam(opts.into()),
            Some(AdamW) => OptimizerDef::AdamW(opts.into()),
        }
    }
}

impl Update<&Options> for OptimizerDef {
    fn update(&mut self, opts: &Options) {
        if let Some(ref optimizer_type) = opts.optimizer {
            if *optimizer_type != self.type_() {
                // If the type is different, re-create the config entirely.
                *self = opts.into();
                return;
            }
        }

        use OptimizerDef::*;
        match self {
            Sgd(ref mut config) => config.update(opts),
            RmsProp(ref mut config) => config.update(opts),
            Adam(ref mut config) => config.update(opts),
            AdamW(ref mut config) => config.update(opts),
        }
    }
}

impl From<&Options> for SgdConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for SgdConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(learning_rate) = opts.learning_rate {
            self.learning_rate = learning_rate;
        }
        if let Some(momentum) = opts.momentum {
            self.momentum = momentum;
        }
        if let Some(weight_decay) = opts.weight_decay {
            self.weight_decay = weight_decay;
        }
    }
}

impl From<&Options> for RmsPropConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for RmsPropConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(learning_rate) = opts.learning_rate {
            self.learning_rate = learning_rate;
        }
        if let Some(momentum) = opts.momentum {
            self.momentum = momentum;
        }
        if let Some(weight_decay) = opts.weight_decay {
            self.weight_decay = weight_decay;
        }
    }
}

impl From<&Options> for AdamConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for AdamConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(learning_rate) = opts.learning_rate {
            self.learning_rate = learning_rate;
        }
        if let Some(weight_decay) = opts.weight_decay {
            self.weight_decay = weight_decay;
        }
    }
}

impl From<&Options> for AdamWConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for AdamWConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(learning_rate) = opts.learning_rate {
            self.learning_rate = learning_rate;
        }
        if let Some(weight_decay) = opts.weight_decay {
            self.weight_decay = weight_decay;
        }
    }
}
