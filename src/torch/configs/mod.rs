//! Module configurations / builders.
mod as_stateful;
mod mlp;
mod rnn;
mod stacked;

pub use mlp::MlpConfig;
pub use rnn::RnnConfig;
pub use stacked::StackedConfig;

/// Configuration for an MLP stacked on top of an RNN.
pub type RnnMlpConfig = StackedConfig<RnnConfig, MlpConfig>;
