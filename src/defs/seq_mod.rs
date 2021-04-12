use crate::torch::configs::{MlpConfig, RnnMlpConfig};
use crate::torch::seq_modules::{GruMlp, LstmMlp, StatefulIterSeqModule, WithState};
use crate::torch::ModuleBuilder;
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
    fn build(
        &self,
        vs: &Path,
        input_dim: usize,
        output_dim: usize,
    ) -> Box<dyn StatefulIterSeqModule> {
        match self {
            SeqModDef::Mlp(config) => {
                let m: Sequential = config.build(vs, input_dim, output_dim);
                Box::new(m)
            }
            SeqModDef::GruMlp(config) => {
                let m: WithState<GruMlp> = config.build(vs, input_dim, output_dim);
                Box::new(m)
            }
            SeqModDef::LstmMlp(config) => {
                let m: WithState<LstmMlp> = config.build(vs, input_dim, output_dim);
                Box::new(m)
            }
        }
    }
}
