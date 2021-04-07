//! Parse environment definition from Options
use super::Options;
use crate::simulation::EnvDef;
use clap::Clap;

/// Environment name
#[derive(Clap, Debug)]
pub enum EnvName {
    SimpleBernoulliBandit,
    BernoulliBandit,
    DeterministicBandit,
    Chain,
}

impl From<&Options> for EnvDef {
    fn from(opts: &Options) -> Self {
        use EnvName::*;
        match opts.environment {
            SimpleBernoulliBandit => EnvDef::SimpleBernoulliBandit,
            BernoulliBandit => EnvDef::BernoulliBandit {
                num_arms: opts.num_actions.unwrap_or(2),
            },
            DeterministicBandit => EnvDef::DeterministicBandit {
                num_arms: opts.num_actions.unwrap_or(2),
            },
            Chain => EnvDef::Chain {
                num_states: opts.num_states,
                discount_factor: opts.discount_factor,
            },
        }
    }
}
