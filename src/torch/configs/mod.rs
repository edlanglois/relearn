//! Module configurations / builders.
mod mlp;
mod rnn;
mod seq_regressor;

pub use mlp::MlpConfig;
pub use rnn::RnnConfig;
pub use seq_regressor::{GruMlpConfig, LstmMlpConfig, SequenceRegressorConfig};
