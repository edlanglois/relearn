//! Serial (single-thread) simulation.
use super::hooks::{BuildSimulationHook, SimulationHook};
use super::{Simulator, SimulatorError, TransientStep};
use crate::agents::{BuildAgent, SynchronousAgent};
use crate::envs::{BuildEnv, Environment};
use crate::logging::{Event, TimeSeriesLogger};
use std::mem;

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

/// Run an agent-environment simulation.
///
/// # Args
/// * `environment` - The environment to simulate.
/// * `agent` - The agent to simulate.
/// * `hook` - A simulation hook run on each step. Controls when the simulation stops.
/// * `logger` - The logger to use.
pub fn run_agent<E, A, H>(
    environment: &mut E,
    agent: &mut A,
    hook: &mut H,
    logger: &mut dyn TimeSeriesLogger,
) where
    E: Environment + ?Sized,
    A: SynchronousAgent<E::Observation, E::Action> + ?Sized,
    H: SimulationHook<E::Observation, E::Action> + ?Sized,
{
    if !hook.start(logger) {
        return;
    }
    let mut observation = environment.reset();
    loop {
        let action = agent.act(&observation);
        let (next, reward) = environment.step(&action, &mut logger.event_logger(Event::EnvStep));

        let mut episode_done = false;
        let (partial_next, next_observation) = next.take_continue_or_else(|| {
            episode_done = true;
            environment.reset()
        });
        let prev_observation = mem::replace(&mut observation, next_observation);

        let step = TransientStep {
            observation: prev_observation,
            action,
            reward,
            next: partial_next.map_continue(|_| &observation),
        };
        let done = !hook.call(&step, logger);
        agent.update(step, logger);
        if episode_done {
            agent.reset();
        }
        if done {
            break;
        }
    }
}
