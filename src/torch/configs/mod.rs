//! Module configurations / builders.
mod as_stateful;
mod mlp;
mod rnn;
mod seq_regressor;

pub use as_stateful::AsStatefulIterConfig;
pub use mlp::MlpConfig;
pub use rnn::RnnConfig;
pub use seq_regressor::{GruMlpConfig, LstmMlpConfig, SequenceRegressorConfig};
