use super::{Options, Update, WithUpdate};
use crate::defs::SeqModDef;
use crate::torch::configs::{MlpConfig, RnnConfig, SequenceRegressorConfig};
use clap::Clap;
use tch::nn::RNN;

/// Sequence module type
#[derive(Clap, Debug, Eq, PartialEq, Clone, Copy)]
pub enum SeqModType {
    Mlp,
    GruMlp,
    LstmMlp,
}

impl SeqModDef {
    pub fn type_(&self) -> SeqModType {
        use SeqModDef::*;
        match self {
            Mlp(_) => SeqModType::Mlp,
            GruMlp(_) => SeqModType::GruMlp,
            LstmMlp(_) => SeqModType::LstmMlp,
        }
    }
}

impl From<&Options> for SeqModDef {
    fn from(opts: &Options) -> Self {
        use SeqModType::*;
        match opts.policy {
            Some(Mlp) | None => SeqModDef::Mlp(opts.into()),
            Some(GruMlp) => SeqModDef::GruMlp(opts.into()),
            Some(LstmMlp) => SeqModDef::LstmMlp(opts.into()),
        }
    }
}

impl Update<&Options> for SeqModDef {
    fn update(&mut self, opts: &Options) {
        if let Some(ref policy_type) = opts.policy {
            if *policy_type != self.type_() {
                // If the type is different, re-create the config entirely.
                *self = opts.into();
                return;
            }
        }

        use SeqModDef::*;
        match self {
            Mlp(ref mut config) => config.update(opts),
            GruMlp(ref mut config) => config.update(opts),
            LstmMlp(ref mut config) => config.update(opts),
        }
    }
}

impl From<&Options> for MlpConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for MlpConfig {
    fn update(&mut self, opts: &Options) {
        if let Some(hidden_sizes) = &opts.hidden_sizes {
            self.hidden_sizes = hidden_sizes.clone();
        }
        if let Some(activation) = opts.activation {
            self.activation = activation;
        }
    }
}

impl<R, P> From<&Options> for SequenceRegressorConfig<R, P>
where
    R: Default + for<'a> Update<&'a Options>,
    P: Default + for<'a> Update<&'a Options>,
{
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<R, P> Update<&Options> for SequenceRegressorConfig<R, P>
where
    R: for<'a> Update<&'a Options>,
    P: for<'a> Update<&'a Options>,
{
    fn update(&mut self, opts: &Options) {
        self.rnn_config.update(opts);
        self.post_config.update(opts);
        if let Some(rnn_hidden_size) = opts.rnn_hidden_size {
            self.rnn_hidden_size = rnn_hidden_size;
        }
        if let Some(rnn_output_activation) = opts.rnn_output_activation {
            self.rnn_output_activation = rnn_output_activation;
        }
    }
}

impl<R: RNN> From<&Options> for RnnConfig<R> {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<R: RNN> Update<&Options> for RnnConfig<R> {
    fn update(&mut self, opts: &Options) {
        if let Some(num_layers) = opts.rnn_num_layers {
            self.num_layers = num_layers
        }
    }
}
