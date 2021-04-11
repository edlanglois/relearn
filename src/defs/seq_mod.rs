use crate::torch::configs::{MlpConfig, RnnMlpConfig};
use crate::torch::seq_modules::{AsStatefulIterator, GruMlp, LstmMlp, StatefulIterSeqModule};
use crate::torch::ModuleBuilder;
use std::borrow::Borrow;
use tch::nn::{Path, Sequential};

/// Sequence module definition
#[derive(Debug)]
pub enum SeqModDef {
    Mlp(MlpConfig),
    GruMlp(RnnMlpConfig),
    LstmMlp(RnnMlpConfig),
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

impl ModuleBuilder<Box<dyn StatefulIterSeqModule>> for SeqModDef {
    fn build<'a, T: Borrow<Path<'a>>>(
        &self,
        vs: T,
        input_dim: usize,
        output_dim: usize,
    ) -> Box<dyn StatefulIterSeqModule> {
        match self {
            SeqModDef::Mlp(config) => {
                let m: Sequential = config.build(vs, input_dim, output_dim);
                Box::new(m)
            }
            SeqModDef::GruMlp(config) => {
                let m: AsStatefulIterator<GruMlp> = config.build(vs, input_dim, output_dim);
                Box::new(m)
            }
            SeqModDef::LstmMlp(config) => {
                let m: AsStatefulIterator<LstmMlp> = config.build(vs, input_dim, output_dim);
                Box::new(m)
            }
        }
    }
}
