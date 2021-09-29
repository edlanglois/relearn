use super::SeqModDef;
use crate::torch::critic::{BuildCritic, Critic, GaeConfig, Return};
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

impl BuildCritic for CriticDef {
    type Critic = Box<dyn Critic + Send>;

    fn build_critic(&self, vs: &Path, in_dim: usize) -> Self::Critic {
        match self {
            CriticDef::Return => Box::new(Return.build_critic(vs, in_dim)),
            CriticDef::Gae(gae_config) => Box::new(gae_config.build_critic(vs, in_dim)),
        }
    }
}
