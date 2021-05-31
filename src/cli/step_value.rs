use super::{Options, Update, WithUpdate};
use crate::defs::StepValueDef;
use crate::torch::step_value::GaeConfig;
use clap::Clap;

/// Agent step value type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Clap)]
pub enum StepValueType {
    Return,
    Gae,
}

impl StepValueDef {
    pub const fn type_(&self) -> StepValueType {
        use StepValueDef::*;
        match self {
            Return => StepValueType::Return,
            Gae(_) => StepValueType::Gae,
        }
    }
}

impl From<&Options> for StepValueDef {
    fn from(opts: &Options) -> Self {
        use StepValueType::*;
        match opts.step_value {
            Some(Return) | None => Self::Return,
            Some(Gae) => Self::Gae(opts.into()),
        }
    }
}

impl Update<&Options> for StepValueDef {
    fn update(&mut self, opts: &Options) {
        use StepValueDef::*;
        if let Some(ref step_value_type) = opts.step_value {
            if *step_value_type != self.type_() {
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
