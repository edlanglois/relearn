//! Parse environment definition from Opts
use super::Opts;
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

impl From<&Opts> for EnvDef {
    fn from(opts: &Opts) -> Self {
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
