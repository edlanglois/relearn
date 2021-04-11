//! Module configurations / builders.
mod as_stateful;
mod mlp;
mod rnn;
mod seq_regressor;

pub use mlp::MlpConfig;
pub use rnn::RnnConfig;
pub use seq_regressor::SequenceRegressorConfig;

/// Configuration for an MLP stacked on top of an RNN.
pub type RnnMlpConfig = SequenceRegressorConfig<RnnConfig, MlpConfig>;
