use super::Opts;
use crate::defs::OptimizerDef;
use crate::torch::optimizers::{AdamConfig, AdamWConfig, RmsPropConfig, SgdConfig};
use clap::Clap;

/// Optimizer name
#[derive(Clap, Debug)]
pub enum OptimizerName {
    Sgd,
    RmsProp,
    Adam,
    AdamW,
}

impl From<&Opts> for OptimizerDef {
    fn from(opts: &Opts) -> Self {
        use OptimizerName::*;
        match opts.optimizer {
            Sgd => OptimizerDef::Sgd(opts.into()),
            RmsProp => OptimizerDef::RmsProp(opts.into()),
            Adam => OptimizerDef::Adam(opts.into()),
            AdamW => OptimizerDef::AdamW(opts.into()),
        }
    }
}

impl From<&Opts> for SgdConfig {
    fn from(opts: &Opts) -> Self {
        let mut config = Self::default();
        if let Some(learning_rate) = opts.learning_rate {
            config.learning_rate = learning_rate;
        }
        if let Some(momentum) = opts.momentum {
            config.momentum = momentum;
        }
        if let Some(weight_decay) = opts.weight_decay {
            config.weight_decay = weight_decay;
        }
        config
    }
}

impl From<&Opts> for RmsPropConfig {
    fn from(opts: &Opts) -> Self {
        let mut config = Self::default();
        if let Some(learning_rate) = opts.learning_rate {
            config.learning_rate = learning_rate;
        }
        if let Some(momentum) = opts.momentum {
            config.momentum = momentum;
        }
        if let Some(weight_decay) = opts.weight_decay {
            config.weight_decay = weight_decay;
        }
        config
    }
}

impl From<&Opts> for AdamConfig {
    fn from(opts: &Opts) -> Self {
        let mut config = Self::default();
        if let Some(learning_rate) = opts.learning_rate {
            config.learning_rate = learning_rate;
        }
        if let Some(weight_decay) = opts.weight_decay {
            config.weight_decay = weight_decay;
        }
        config
    }
}

impl From<&Opts> for AdamWConfig {
    fn from(opts: &Opts) -> Self {
        let mut config = Self::default();
        if let Some(learning_rate) = opts.learning_rate {
            config.learning_rate = learning_rate;
        }
        if let Some(weight_decay) = opts.weight_decay {
            config.weight_decay = weight_decay;
        }
        config
    }
}
