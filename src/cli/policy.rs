use super::Options;
use crate::defs::PolicyDef;
use crate::torch::configs::{MlpConfig, RnnConfig, SequenceRegressorConfig};
use clap::Clap;
use tch::nn::RNN;

/// Policy name
#[derive(Clap, Debug)]
pub enum PolicyName {
    Mlp,
    GruMlp,
    LstmMlp,
}

impl From<&Options> for PolicyDef {
    fn from(opts: &Options) -> Self {
        use PolicyName::*;
        match opts.policy {
            Mlp => PolicyDef::Mlp(opts.into()),
            GruMlp => PolicyDef::GruMlp(opts.into()),
            LstmMlp => PolicyDef::LstmMlp(opts.into()),
        }
    }
}

impl From<&Options> for MlpConfig {
    fn from(opts: &Options) -> Self {
        let mut config = MlpConfig::default();
        if let Some(hidden_sizes) = &opts.hidden_sizes {
            config.hidden_sizes = hidden_sizes.clone();
        }
        if let Some(activation) = opts.activation {
            config.activation = activation;
        }
        config
    }
}

impl<R, P> From<&Options> for SequenceRegressorConfig<R, P>
where
    R: Default,
    P: Default,
    for<'a> &'a Options: Into<R> + Into<P>,
{
    fn from(opts: &Options) -> Self {
        let mut config = Self::default();
        config.rnn_config = opts.into();
        config.post_config = opts.into();
        if let Some(rnn_hidden_size) = opts.rnn_hidden_size {
            config.rnn_hidden_size = rnn_hidden_size;
        }
        if let Some(rnn_output_activation) = opts.rnn_output_activation {
            config.rnn_output_activation = rnn_output_activation;
        }
        config
    }
}

impl<R: RNN> From<&Options> for RnnConfig<R> {
    fn from(opts: &Options) -> Self {
        let mut config = RnnConfig::default();
        if let Some(num_layers) = opts.rnn_num_layers {
            config.num_layers = num_layers
        }
        config
    }
}
