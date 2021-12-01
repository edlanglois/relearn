//! Serial (single-thread) simulation.
use super::hooks::BuildSimulationHook;
use super::{run_agent, Simulator, SimulatorError};
use crate::agents::BuildAgent;
use crate::envs::BuildEnv;
use crate::logging::TimeSeriesLogger;

/// Serial (single-thread) simulator.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SerialSimulator<EC, AC, HC> {
    env_config: EC,
    agent_config: AC,
    hook_config: HC,
}

impl<EC, AC, HC> SerialSimulator<EC, AC, HC> {
    pub const fn new(env_config: EC, agent_config: AC, hook_config: HC) -> Self {
        Self {
            env_config,
            agent_config,
            hook_config,
        }
    }
}

impl<EC, AC, HC> Simulator for SerialSimulator<EC, AC, HC>
where
    EC: BuildEnv,
    EC::Observation: Clone,
    AC: BuildAgent<EC::ObservationSpace, EC::ActionSpace>,
    HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace>,
{
    fn run_simulation(
        &self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), SimulatorError> {
        let mut env = self.env_config.build_env(env_seed)?;
        let mut agent = self.agent_config.build_agent(&env, agent_seed)?;
        let mut hook = self.hook_config.build_hook(&env, 1, 0);
        run_agent(&mut env, &mut agent, &mut hook, logger);
        Ok(())
    }
}
