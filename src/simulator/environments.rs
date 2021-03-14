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
    BernoulliBandit { means: Vec<f32> },
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
                let env = BernoulliBandit::new(vec![0.2, 0.8], seed);
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(TypedSimulator::new(Box::new(env), agent, logger)))
            }
            EnvDef::BernoulliBandit { means } => {
                let env = BernoulliBandit::new(means, seed);
                let agent = agent_def.make_finite_finite(env.structure(), seed + 1)?;
                Ok(Box::new(TypedSimulator::new(Box::new(env), agent, logger)))
            }
        }
    }
}
