//! Environment definitions
use super::{AgentDef, BoxedSimulator, MakeAgentError, Simulation};
use crate::envs::{
    AsStateful, BernoulliBandit, Chain, DeterministicBandit, Environment, StatefulEnvironment,
};
use crate::logging::Logger;
use rand::prelude::*;

/// The definition of an environment
#[derive(Debug)]
pub enum EnvDef {
    /// A BernoulliBandit with means [0.2, 0.8]
    SimpleBernoulliBandit,
    /// A BernoulliBandit with means uniformly sampled from [0, 1]
    BernoulliBandit { num_arms: u32 },
    /// A DeterministicBandit with values uniformly sampled from [0, 1]
    DeterministicBandit { num_arms: u32 },
    /// The Chain environment
    Chain {
        num_states: Option<u32>,
        discount_factor: Option<f32>,
    },
}

impl EnvDef {
    pub fn make_simulation<L: Logger + 'static>(
        self,
        agent_def: AgentDef,
        seed: u64,
        logger: L,
    ) -> Result<Box<dyn Simulation>, MakeAgentError> {
        match self {
            EnvDef::SimpleBernoulliBandit => {
                let env = BernoulliBandit::from_means(vec![0.2, 0.8]).unwrap();
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(BoxedSimulator::new(
                    Box::new(env.as_stateful(seed)),
                    agent,
                    logger,
                )))
            }
            EnvDef::BernoulliBandit { num_arms } => {
                let env = BernoulliBandit::uniform(num_arms, &mut StdRng::seed_from_u64(seed + 2));
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(BoxedSimulator::new(
                    Box::new(env.as_stateful(seed)),
                    agent,
                    logger,
                )))
            }
            EnvDef::DeterministicBandit { num_arms } => {
                let mut rng = StdRng::seed_from_u64(seed + 2);
                let env =
                    DeterministicBandit::from_values((0..num_arms).into_iter().map(|_| rng.gen()));
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(BoxedSimulator::new(
                    Box::new(env.as_stateful(seed)),
                    agent,
                    logger,
                )))
            }
            EnvDef::Chain {
                num_states,
                discount_factor,
            } => {
                let env = Chain::new(num_states, discount_factor).as_stateful(seed);
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(BoxedSimulator::new(Box::new(env), agent, logger)))
            }
        }
    }
}
