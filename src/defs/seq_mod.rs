use crate::torch::configs::{GruMlpConfig, LstmMlpConfig, MlpConfig};
use crate::torch::seq_modules::{AsStatefulIterator, StatefulIterSeqModule};
use crate::torch::ModuleBuilder;
use std::borrow::Borrow;
use tch::nn::Path;

/// Sequence module definition
#[derive(Debug)]
pub enum SeqModDef {
    Mlp(MlpConfig),
    GruMlp(GruMlpConfig),
    LstmMlp(LstmMlpConfig),
}

impl Default for SeqModDef {
    fn default() -> Self {
        SeqModDef::Mlp(Default::default())
    }
}

impl From<MlpConfig> for SeqModDef {
    fn from(c: MlpConfig) -> Self {
        SeqModDef::Mlp(c)
    }
}

impl From<GruMlpConfig> for SeqModDef {
    fn from(c: GruMlpConfig) -> Self {
        SeqModDef::GruMlp(c)
    }
}

impl From<LstmMlpConfig> for SeqModDef {
    fn from(c: LstmMlpConfig) -> Self {
        SeqModDef::LstmMlp(c)
    }
}

impl ModuleBuilder for SeqModDef {
    type Module = Box<dyn StatefulIterSeqModule>;
    fn build<'a, T: Borrow<Path<'a>>>(
        &self,
        vs: T,
        input_dim: usize,
        output_dim: usize,
    ) -> Self::Module {
        use SeqModDef::*;
        match self {
            Mlp(config) => Box::new(config.build(vs, input_dim, output_dim)),
            GruMlp(config) => Box::new(AsStatefulIterator::from(
                config.build(vs, input_dim, output_dim),
            )),
            LstmMlp(config) => Box::new(AsStatefulIterator::from(
                config.build(vs, input_dim, output_dim),
            )),
        }
    }
}
