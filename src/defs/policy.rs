use crate::torch::configs::{GruMlpConfig, LstmMlpConfig, MlpConfig};

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
