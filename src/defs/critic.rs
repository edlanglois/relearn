use super::SeqModDef;
use crate::torch::critic::{Critic, BuildCritic, Gae, GaeConfig, Return};
use crate::torch::seq_modules::StatefulIterSeqModule;
use tch::nn::Path;

/// Critic definition
#[derive(Debug, Clone, PartialEq)]
pub enum CriticDef {
    Return,
    Gae(GaeConfig<SeqModDef>),
}

impl Default for CriticDef {
    fn default() -> Self {
        Self::Return
    }
}

impl BuildCritic<Box<dyn Critic>> for CriticDef {
    fn build_critic(&self, vs: &Path, in_dim: usize) -> Box<dyn Critic> {
        match self {
            CriticDef::Return => Box::new(Return.build_critic(vs, in_dim)),
            CriticDef::Gae(gae_config) => {
                let gae: Gae<Box<dyn StatefulIterSeqModule>> = gae_config.build_critic(vs, in_dim);
                Box::new(gae)
            }
        }
    }
}
