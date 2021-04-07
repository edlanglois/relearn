//! Environment definitions
use super::hooks::{GenericSimulationHook, SimulationHook, StepLogger};
use super::{BoxedSimulator, Simulation};
use crate::agents::NewAgentError;
use crate::defs::AgentDef;
use crate::envs::{AsStateful, BernoulliBandit, Chain, DeterministicBandit, StatefulEnvironment};
use crate::logging::Logger;
use crate::spaces::{FiniteSpace, RLSpace, Space};
use rand::prelude::*;
use std::fmt::Debug;

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
        discount_factor: Option<f64>,
    },
}

fn finite_finite_simulator<OS, AS, L, H>(
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent_def: AgentDef,
    logger: L,
    hook: H,
    seed: u64,
) -> Result<Box<dyn Simulation>, NewAgentError>
where
    OS: RLSpace + FiniteSpace + Clone + 'static,
    <OS as Space>::Element: Clone,
    AS: RLSpace + FiniteSpace + Clone + 'static,
    L: Logger + 'static,
    H: SimulationHook<<OS as Space>::Element, <AS as Space>::Element, L> + 'static,
{
    let env_structure = environment.structure();
    let log_hook = StepLogger::new(
        env_structure.observation_space.clone(),
        env_structure.action_space.clone(),
    );
    let hook = (log_hook, hook);
    let agent = agent_def.build_finite_finite(env_structure, seed)?;
    Ok(Box::new(BoxedSimulator::new(
        environment,
        agent,
        logger,
        hook,
    )))
}

impl EnvDef {
    pub fn make_simulation<L: Logger + 'static, H: GenericSimulationHook + 'static>(
        self,
        agent_def: AgentDef,
        seed: u64,
        logger: L,
        hook: H,
    ) -> Result<Box<dyn Simulation>, NewAgentError> {
        match self {
            EnvDef::SimpleBernoulliBandit => {
                let env = BernoulliBandit::from_means(vec![0.2, 0.8])
                    .unwrap()
                    .as_stateful(seed);
                finite_finite_simulator(Box::new(env), agent_def, logger, hook, seed + 1)
            }
            EnvDef::BernoulliBandit { num_arms } => {
                let env = BernoulliBandit::uniform(num_arms, &mut StdRng::seed_from_u64(seed + 2))
                    .as_stateful(seed);
                finite_finite_simulator(Box::new(env), agent_def, logger, hook, seed + 1)
            }
            EnvDef::DeterministicBandit { num_arms } => {
                let mut rng = StdRng::seed_from_u64(seed + 2);
                let env = DeterministicBandit::from_values(
                    (0..num_arms).into_iter().map(|_| rng.gen::<f64>()),
                )
                .as_stateful(seed);
                finite_finite_simulator(Box::new(env), agent_def, logger, hook, seed + 1)
            }
            EnvDef::Chain {
                num_states,
                discount_factor,
            } => {
                let mut env = Chain::default();
                if let Some(num_states) = num_states {
                    env.size = num_states;
                }
                if let Some(discount_factor) = discount_factor {
                    env.discount_factor = discount_factor;
                }
                let env = env.as_stateful(seed);
                finite_finite_simulator(Box::new(env), agent_def, logger, hook, seed + 1)
            }
        }
    }
}
