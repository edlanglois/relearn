use crate::torch::{
    modules::MlpConfig,
    policy::Policy,
    seq_modules::{
        GruConfig, IterativeModule, LstmConfig, SequenceModule, StackedConfig, WithState,
    },
    BuildModule,
};
use tch::nn::Path;

/// Sequence module definition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SeqModDef {
    Mlp(MlpConfig),
    GruMlp(StackedConfig<GruConfig, MlpConfig>),
    LstmMlp(StackedConfig<LstmConfig, MlpConfig>),
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

impl BuildModule for SeqModDef {
    type Module = Box<dyn SequenceModule + Send>;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        match self {
            SeqModDef::Mlp(config) => Box::new(config.build_module(vs, in_dim, out_dim)),
            SeqModDef::GruMlp(config) => Box::new(config.build_module(vs, in_dim, out_dim)),
            SeqModDef::LstmMlp(config) => Box::new(config.build_module(vs, in_dim, out_dim)),
        }
    }
}

/// Policy module definition
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PolicyDef(pub SeqModDef);

impl From<SeqModDef> for PolicyDef {
    fn from(config: SeqModDef) -> Self {
        Self(config)
    }
}

fn build_module_with_state<T>(
    config: &T,
    vs: &Path,
    in_dim: usize,
    out_dim: usize,
) -> WithState<T::Module>
where
    T: BuildModule + ?Sized,
    <T as BuildModule>::Module: IterativeModule,
{
    config.build_module(vs, in_dim, out_dim).into()
}

impl BuildModule for PolicyDef {
    type Module = Box<dyn Policy + Send>;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        match &self.0 {
            SeqModDef::Mlp(config) => {
                Box::new(build_module_with_state(config, vs, in_dim, out_dim))
            }
            SeqModDef::GruMlp(config) => {
                Box::new(build_module_with_state(config, vs, in_dim, out_dim))
            }
            SeqModDef::LstmMlp(config) => {
                Box::new(build_module_with_state(config, vs, in_dim, out_dim))
            }
        }
    }
}
