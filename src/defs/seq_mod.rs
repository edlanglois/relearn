use crate::torch::{
    agents::TrpoPolicyModule,
    modules::MlpConfig,
    seq_modules::{GruMlp, LstmMlp, RnnMlpConfig, StatefulIterSeqModule, WithState},
    ModuleBuilder,
};
use tch::nn::{Path, Sequential};

/// Sequence module definition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SeqModDef {
    Mlp(MlpConfig),
    GruMlp(RnnMlpConfig),
    LstmMlp(RnnMlpConfig),
}

impl Default for SeqModDef {
    fn default() -> Self {
        Self::Mlp(MlpConfig::default())
    }
}

impl From<MlpConfig> for SeqModDef {
    fn from(c: MlpConfig) -> Self {
        Self::Mlp(c)
    }
}

// TODO: Make generic once std::marker::Unsize is stabilized:
// impl<T> ModuleBuilder<Box<T>> for SeqModDef
// where
//      Sequential: Unsize<T>,
//      WithState<GruMlp>: Unsize<T>,
//      WithState<LstmMlp>: Usize<T>,
// {
//      ...
//          Box::new(m) as Box<T>
//      ...
// }

macro_rules! boxed_module_builder_for_seq_mod {
    ($type:ty) => {
        impl ModuleBuilder<Box<$type>> for SeqModDef {
            fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Box<$type> {
                match self {
                    SeqModDef::Mlp(config) => {
                        let m: Sequential = config.build_module(vs, in_dim, out_dim);
                        Box::new(m)
                    }
                    SeqModDef::GruMlp(config) => {
                        let m: WithState<GruMlp> = config.build_module(vs, in_dim, out_dim);
                        Box::new(m)
                    }
                    SeqModDef::LstmMlp(config) => {
                        let m: WithState<LstmMlp> = config.build_module(vs, in_dim, out_dim);
                        Box::new(m)
                    }
                }
            }
        }
    };
}

boxed_module_builder_for_seq_mod!(dyn StatefulIterSeqModule);
boxed_module_builder_for_seq_mod!(dyn TrpoPolicyModule);
