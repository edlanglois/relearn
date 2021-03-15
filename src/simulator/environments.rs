//! Environment definitions
use super::{AgentDef, MakeAgentError, Simulator, TypedSimulator};
use crate::envs::{BernoulliBandit, StructuredEnvironment};
use crate::loggers::Logger;

/// The definition of an environment
#[derive(Debug)]
pub enum EnvDef {
    /// A BernoulliBandit with means [0.2, 0.8]
    SimpleBernoulliBandit,
    /// A BernoulliBandit
    BernoulliBandit { num_arms: usize },
}

impl EnvDef {
    pub fn make_simulator<L: Logger + 'static>(
        self,
        agent_def: AgentDef,
        seed: u64,
        logger: L,
    ) -> Result<Box<dyn Simulator>, MakeAgentError> {
        match self {
            EnvDef::SimpleBernoulliBandit => {
                let env = BernoulliBandit::from_means(vec![0.2, 0.8], seed);
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(TypedSimulator::new(Box::new(env), agent, logger)))
            }
            EnvDef::BernoulliBandit { num_arms } => {
                let env = BernoulliBandit::uniform(num_arms, seed);
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(TypedSimulator::new(Box::new(env), agent, logger)))
            }
        }
    }
}
