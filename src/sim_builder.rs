use crate::agents::{Agent, RandomAgent, TabularQLearningAgent};
use crate::envs::{BernoulliBandit, EnvSpec, Environment};
use crate::spaces::{FiniteSpace, Space};

use std::fmt::Debug;

pub fn make_simulator(env_def: EnvDefs, agent_def: AgentDefs, seed: u64) -> Box<dyn Simulator> {
    match env_def {
        EnvDefs::SimpleBernoulliBandit => {
            let env = BernoulliBandit::new(vec![0.2, 0.8], seed);
            let agent = agent_def.make_finite_finite(env.env_spec(), seed + 1);
            Box::new(TypedSimulator {
                environment: Box::new(env),
                agent,
            })
        }
    }
}
