use super::{Options, Update, WithUpdate};
use crate::defs::CriticDef;
use crate::torch::critic::GaeConfig;
use clap::ArgEnum;

/// Agent step value type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum CriticType {
    Return,
    Gae,
}

impl CriticDef {
    pub const fn type_(&self) -> CriticType {
        use CriticDef::*;
        match self {
            Return => CriticType::Return,
            Gae(_) => CriticType::Gae,
        }
    }
}

impl From<&Options> for CriticDef {
    fn from(opts: &Options) -> Self {
        use CriticType::*;
        match opts.critic {
            Some(Return) | None => Self::Return,
            Some(Gae) => Self::Gae(opts.into()),
        }
    }
}

impl Update<&Options> for CriticDef {
    fn update(&mut self, opts: &Options) {
        use CriticDef::*;
        if let Some(ref critic_type) = opts.critic {
            if *critic_type != self.type_() {
                // If the type is different, re-create the config entirely.
                *self = opts.into();
                return;
            }
        }

        match self {
            Return => {}
            Gae(config) => config.update(opts),
        }
    }
}

impl<'a, T> From<&'a Options> for GaeConfig<T>
where
    T: Default + Update<&'a Options>,
{
    fn from(opts: &'a Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl<'a, T> Update<&'a Options> for GaeConfig<T>
where
    T: Default + Update<&'a Options>,
{
    fn update(&mut self, opts: &'a Options) {
        // TODO: Distinguish policy/value fn options
        self.value_fn_config.update(opts);
        if let Some(discount_factor) = opts.gae_discount_factor {
            self.gamma = discount_factor;
        }
        if let Some(lambda) = opts.gae_lambda {
            self.lambda = lambda;
        }
    }
}
