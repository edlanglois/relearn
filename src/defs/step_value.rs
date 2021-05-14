use super::SeqModDef;
use crate::torch::seq_modules::StatefulIterSeqModule;
use crate::torch::step_value::{Gae, GaeConfig, Return, StepValue, StepValueBuilder};
use tch::nn::Path;

/// Step value definition
#[derive(Debug)]
pub enum StepValueDef {
    Return,
    Gae(GaeConfig<SeqModDef>),
}

impl Default for StepValueDef {
    fn default() -> Self {
        Self::Return
    }
}

impl StepValueBuilder<Box<dyn StepValue>> for StepValueDef {
    fn build_step_value(&self, vs: &Path, in_dim: usize) -> Box<dyn StepValue> {
        match self {
            StepValueDef::Return => Box::new(Return.build_step_value(vs, in_dim)),
            StepValueDef::Gae(gae_config) => {
                let gae: Gae<Box<dyn StatefulIterSeqModule>> =
                    gae_config.build_step_value(vs, in_dim);
                Box::new(gae)
            }
        }
    }
}
