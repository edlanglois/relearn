//! Serial (single-thread) simulation.
use super::hooks::{BuildSimulationHook, SimulationHook};
use super::{Simulator, SimulatorError, TransientStep};
use crate::agents::{Actor, BuildAgent, SynchronousAgent};
use crate::envs::{BuildEnv, Environment, Successor};
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
        let mut reset = || {
            episode_done = true;
            environment.reset()
        };

        let (partial_successor, next_observation) = match next {
            Successor::Continue(next_obs) => (Successor::Continue(()), next_obs),
            Successor::Terminate => (Successor::Terminate, reset()),
            Successor::Interrupt(next_obs) => (Successor::Interrupt(next_obs), reset()),
        };
        let prev_observation = mem::replace(&mut observation, next_observation);
        let ref_successor = match partial_successor {
            Successor::Continue(()) => Successor::Continue(&observation),
            Successor::Terminate => Successor::Terminate,
            Successor::Interrupt(next_obs) => Successor::Interrupt(next_obs),
        };

        let step = TransientStep {
            observation: prev_observation,
            action,
            reward,
            next: ref_successor,
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

/// Run an actor-environment simulation without reward feedback.
///
/// An actor may depend on history from the current episode.
///
/// # Args
/// * `environment` - The environment to simulate.
/// * `actor` - The actor to simulate.
/// * `hook` - A simulation hook run on each step. Controls when the simulation stops.
/// * `logger` - The logger to use. Passed to hook calls.
pub fn run_actor<E, A, H>(
    environment: &mut E,
    actor: &mut A,
    hook: &mut H,
    logger: &mut dyn TimeSeriesLogger,
) where
    E: Environment + ?Sized,
    A: Actor<E::Observation, E::Action> + ?Sized,
    H: SimulationHook<E::Observation, E::Action> + ?Sized,
{
    if !hook.start(logger) {
        return;
    }
    let mut observation = environment.reset();
    loop {
        let action = actor.act(&observation);
        let (next, reward) = environment.step(&action, &mut logger.event_logger(Event::EnvStep));

        let mut reset = || {
            actor.reset();
            environment.reset()
        };

        let (partial_successor, next_observation) = match next {
            Successor::Continue(next_obs) => (Successor::Continue(()), next_obs),
            Successor::Terminate => (Successor::Terminate, reset()),
            Successor::Interrupt(next_obs) => (Successor::Interrupt(next_obs), reset()),
        };
        let prev_observation = mem::replace(&mut observation, next_observation);
        let ref_successor = match partial_successor {
            Successor::Continue(()) => Successor::Continue(&observation),
            Successor::Terminate => Successor::Terminate,
            Successor::Interrupt(next_obs) => Successor::Interrupt(next_obs),
        };

        let step = TransientStep {
            observation: prev_observation,
            action,
            reward,
            next: ref_successor,
        };
        if !hook.call(&step, logger) {
            break;
        }
    }
}
