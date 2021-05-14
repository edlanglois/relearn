use super::{Options, Update, WithUpdate};
use crate::defs::SeqModDef;
use crate::torch::modules::MlpConfig;
use crate::torch::seq_modules::{RnnConfig, StackedConfig};
use clap::Clap;

/// Sequence module type
#[derive(Clap, Debug, Eq, PartialEq, Clone, Copy)]
pub enum SeqModType {
    Mlp,
    GruMlp,
    LstmMlp,
}

impl SeqModDef {
    pub const fn type_(&self) -> SeqModType {
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
            Some(Mlp) | None => Self::Mlp(opts.into()),
            Some(GruMlp) => Self::GruMlp(opts.into()),
            Some(LstmMlp) => Self::LstmMlp(opts.into()),
        }
    }
}

impl Update<&Options> for SeqModDef {
    fn update(&mut self, opts: &Options) {
        use SeqModDef::*;
        if let Some(ref policy_type) = opts.policy {
            if *policy_type != self.type_() {
                // If the type is different, re-create the config entirely.
                *self = opts.into();
                return;
            }
        }

        match self {
            Mlp(ref mut config) => config.update(opts),
            GruMlp(ref mut config) | LstmMlp(ref mut config) => config.update(opts),
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

impl<'a, R, P> From<&'a Options> for StackedConfig<R, P>
where
    R: Default + Update<&'a Options>,
    P: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, R, P> Update<&'a Options> for StackedConfig<R, P>
where
    R: Update<&'a Options>,
    P: Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        self.seq_config.update(opts);
        self.top_config.update(opts);
        if let Some(seq_output_dim) = opts.rnn_hidden_size {
            self.seq_output_dim = seq_output_dim;
        }
        if let Some(seq_output_activation) = opts.rnn_output_activation {
            self.seq_output_activation = seq_output_activation;
        }
    }
}

impl From<&Options> for RnnConfig {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for RnnConfig {
    fn update(&mut self, _opts: &Options) {}
}
