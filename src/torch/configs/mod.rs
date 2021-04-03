//! Module configurations / builders.
mod mlp;
mod rnn;
mod seq_regressor;

pub use mlp::MLPConfig;
pub use rnn::RNNConfig;
pub use seq_regressor::{GruMlpConfig, LstmMlpConfig, SequenceRegressorConfig};
