//! Serial (single-thread) simulation.
use super::hooks::SimulationHook;
use super::Simulator;
use crate::agents::{Actor, Agent, BuildAgent, Step};
use crate::envs::{BuildEnv, Environment};
use crate::error::RLError;
use crate::logging::TimeSeriesLogger;

/// Serial (single-thread) simulator.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SerialSimulator<EC, AC, H> {
    env_config: EC,
    agent_config: AC,
    hook: H,
}

impl<EC, AC, H> SerialSimulator<EC, AC, H> {
    pub const fn new(env_config: EC, agent_config: AC, hook: H) -> Self {
        Self {
            env_config,
            agent_config,
            hook,
        }
    }
}

impl<EC, AC, H> Simulator for SerialSimulator<EC, AC, H>
where
    EC: BuildEnv,
    EC::Observation: Clone,
    AC: BuildAgent<EC::ObservationSpace, EC::ActionSpace>,
    H: SimulationHook<EC::Observation, EC::Action> + Clone,
{
    fn run_simulation(
        &mut self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), RLError> {
        let mut env = self.env_config.build_env(env_seed)?;
        let mut agent = self.agent_config.build_agent(&env, agent_seed)?;
        let mut hook = self.hook.clone();
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
    // The ?Sized allows this function to be called with types
    // (&mut dyn Environment, &mut dyn Agent, ...
    // In that case it only needs to be instantiated once and can work with trait pointers.
    //
    // Alternatively, it can use the concrete struct types, which allows inlining.
    E: Environment + ?Sized,
    E::Observation: Clone,
    A: Agent<E::Observation, E::Action> + ?Sized,
    H: SimulationHook<E::Observation, E::Action> + ?Sized,
{
    if !hook.start(logger) {
        return;
    }
    let mut observation = environment.reset();
    let mut new_episode = true;

    loop {
        let action = agent.act(&observation, new_episode);
        let (next_observation, reward, episode_done) = environment.step(&action);

        new_episode = false;
        let step = Step {
            observation,
            action,
            reward,
            next_observation: next_observation.clone(),
            episode_done,
        };

        let stop_simulation = !hook.call(&step, logger);
        agent.update(step, logger);
        if stop_simulation {
            break;
        }

        if episode_done {
            new_episode = true;
            observation = environment.reset();
        } else {
            observation =
                next_observation.expect("Observation must exist if the episode is not done")
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
    E::Observation: Clone,
    A: Actor<E::Observation, E::Action> + ?Sized,
    H: SimulationHook<E::Observation, E::Action> + ?Sized,
{
    if !hook.start(logger) {
        return;
    }
    let mut observation = environment.reset();
    let mut new_episode = true;

    loop {
        let action = actor.act(&observation, new_episode);
        let (next_observation, reward, episode_done) = environment.step(&action);

        new_episode = false;
        let step = Step {
            observation,
            action,
            reward,
            next_observation: next_observation.clone(),
            episode_done,
        };

        if !hook.call(&step, logger) {
            break;
        }

        observation = if episode_done {
            environment.reset()
        } else {
            next_observation.expect("Observation must exist if the episode is not done")
        };
    }
}
