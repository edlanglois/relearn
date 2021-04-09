use crate::torch::configs::{GruMlpConfig, LstmMlpConfig, MlpConfig};
use crate::torch::seq_modules::{AsStatefulIterator, StatefulIterSeqModule};
use crate::torch::ModuleBuilder;
use std::borrow::Borrow;
use tch::nn::Path;

/// Torch policy definition.
#[derive(Debug)]
pub enum PolicyDef {
    Mlp(MlpConfig),
    GruMlp(GruMlpConfig),
    LstmMlp(LstmMlpConfig),
}

impl Default for PolicyDef {
    fn default() -> Self {
        PolicyDef::Mlp(Default::default())
    }
}

impl From<MlpConfig> for PolicyDef {
    fn from(c: MlpConfig) -> Self {
        PolicyDef::Mlp(c)
    }
}

impl From<GruMlpConfig> for PolicyDef {
    fn from(c: GruMlpConfig) -> Self {
        PolicyDef::GruMlp(c)
    }
}

impl From<LstmMlpConfig> for PolicyDef {
    fn from(c: LstmMlpConfig) -> Self {
        PolicyDef::LstmMlp(c)
    }
}

impl ModuleBuilder for PolicyDef {
    type Module = Box<dyn StatefulIterSeqModule>;
    fn build<'a, T: Borrow<Path<'a>>>(
        &self,
        vs: T,
        input_dim: usize,
        output_dim: usize,
    ) -> Self::Module {
        use PolicyDef::*;
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
