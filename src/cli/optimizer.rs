use super::{Update, WithUpdate};
use crate::defs::OptimizerDef;
use crate::torch::optimizers::{AdamConfig, AdamWConfig, RmsPropConfig, SgdConfig};
use clap::Clap;

/// Optimizer name
#[derive(Clap, Debug, PartialEq, Eq, Clone, Copy)]
pub enum OptimizerType {
    Sgd,
    RmsProp,
    Adam,
    AdamW,
}

/// Optimizer options
pub trait OptimizerOptions {
    fn type_(&self) -> Option<OptimizerType>;
    fn learning_rate(&self) -> Option<f64>;
    fn momentum(&self) -> Option<f64>;
    fn weight_decay(&self) -> Option<f64>;
}

impl OptimizerDef {
    pub const fn type_(&self) -> OptimizerType {
        use OptimizerDef::*;
        match self {
            Sgd(_) => OptimizerType::Sgd,
            RmsProp(_) => OptimizerType::RmsProp,
            Adam(_) => OptimizerType::Adam,
            AdamW(_) => OptimizerType::AdamW,
        }
    }
}

impl<T: OptimizerOptions> From<&T> for OptimizerDef {
    fn from(opts: &T) -> Self {
        use OptimizerType::*;
        match opts.type_() {
            Some(Sgd) => Self::Sgd(opts.into()),
            Some(RmsProp) => Self::RmsProp(opts.into()),
            Some(Adam) | None => Self::Adam(opts.into()),
            Some(AdamW) => Self::AdamW(opts.into()),
        }
    }
}

impl<T: OptimizerOptions> Update<&T> for OptimizerDef {
    fn update(&mut self, opts: &T) {
        use OptimizerDef::*;

        if let Some(ref optimizer_type) = opts.type_() {
            if *optimizer_type != self.type_() {
                // If the type is different, re-create the config entirely.
                *self = opts.into();
                return;
            }
        }
        match self {
            Sgd(ref mut config) => config.update(opts),
            RmsProp(ref mut config) => config.update(opts),
            Adam(ref mut config) => config.update(opts),
            AdamW(ref mut config) => config.update(opts),
        }
    }
}

impl<T: OptimizerOptions> From<&T> for SgdConfig {
    fn from(opts: &T) -> Self {
        Self::default().with_update(opts)
    }
}

impl<T: OptimizerOptions> Update<&T> for SgdConfig {
    fn update(&mut self, opts: &T) {
        if let Some(learning_rate) = opts.learning_rate() {
            self.learning_rate = learning_rate;
        }
        if let Some(momentum) = opts.momentum() {
            self.momentum = momentum;
        }
        if let Some(weight_decay) = opts.weight_decay() {
            self.weight_decay = weight_decay;
        }
    }
}

impl<T: OptimizerOptions> From<&T> for RmsPropConfig {
    fn from(opts: &T) -> Self {
        Self::default().with_update(opts)
    }
}

impl<T: OptimizerOptions> Update<&T> for RmsPropConfig {
    fn update(&mut self, opts: &T) {
        if let Some(learning_rate) = opts.learning_rate() {
            self.learning_rate = learning_rate;
        }
        if let Some(momentum) = opts.momentum() {
            self.momentum = momentum;
        }
        if let Some(weight_decay) = opts.weight_decay() {
            self.weight_decay = weight_decay;
        }
    }
}

impl<T: OptimizerOptions> From<&T> for AdamConfig {
    fn from(opts: &T) -> Self {
        Self::default().with_update(opts)
    }
}

impl<T: OptimizerOptions> Update<&T> for AdamConfig {
    fn update(&mut self, opts: &T) {
        if let Some(learning_rate) = opts.learning_rate() {
            self.learning_rate = learning_rate;
        }
        if let Some(weight_decay) = opts.weight_decay() {
            self.weight_decay = weight_decay;
        }
    }
}

impl<T: OptimizerOptions> From<&T> for AdamWConfig {
    fn from(opts: &T) -> Self {
        Self::default().with_update(opts)
    }
}

impl<T: OptimizerOptions> Update<&T> for AdamWConfig {
    fn update(&mut self, opts: &T) {
        if let Some(learning_rate) = opts.learning_rate() {
            self.learning_rate = learning_rate;
        }
        if let Some(weight_decay) = opts.weight_decay() {
            self.weight_decay = weight_decay;
        }
    }
}
