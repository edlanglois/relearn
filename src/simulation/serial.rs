//! Serial (single-thread) simulation.
use super::hooks::SimulationHook;
use super::RunSimulation;
use crate::agents::{Actor, Agent, Step};
use crate::envs::{EnvStructure, StatefulEnvironment};
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;

/// An agent-environment simulator with logging.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Simulator<E, A, H, L> {
    environment: E,
    agent: A,
    logger: L,
    hook: H,
}

impl<E, A, H, L> Simulator<E, A, H, L> {
    pub const fn new(environment: E, agent: A, hook: H, logger: L) -> Self {
        Self {
            environment,
            agent,
            hook,
            logger,
        }
    }
}

impl<E, A, H, L> RunSimulation for Simulator<E, A, H, L>
where
    E: StatefulEnvironment,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
        <<E as EnvStructure>::ObservationSpace as Space>::Element,
        <<E as EnvStructure>::ActionSpace as Space>::Element,
    >,
    H: SimulationHook<
        <<E as EnvStructure>::ObservationSpace as Space>::Element,
        <<E as EnvStructure>::ActionSpace as Space>::Element,
    >,
    L: TimeSeriesLogger,
{
    fn run_simulation(&mut self) {
        run_agent(
            &mut self.environment,
            &mut self.agent,
            &mut self.hook,
            &mut self.logger,
        );
    }
}

/// Run an agent-environment simulation.
///
/// # Args
/// * `environment` - The environment to simulate.
/// * `agent` - The agent to simulate.
/// * `hook` - A simulation hook run on each step. Controls when the simulation stops.
/// * `logger` - The logger to use.
pub fn run_agent<E, A, H, L>(environment: &mut E, agent: &mut A, hook: &mut H, logger: &mut L)
where
    // The ?Sized allows this function to be called with types
    // (&mut dyn Environment, &mut dyn Agent, ...
    // In that case it only needs to be instantiated once and can work with trait pointers.
    //
    // Alternatively, it can use the concrete struct types, which allows inlining.
    E: StatefulEnvironment + ?Sized,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + ?Sized,
    H: SimulationHook<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + ?Sized,
    // Not ?Sized because can't convert &(TimeSeriesLogger + ?Sized) => &mut dyn TimeSeriesLogger
    L: TimeSeriesLogger,
{
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
pub fn run_actor<E, A, H, L>(environment: &mut E, actor: &mut A, hook: &mut H, logger: &mut L)
where
    E: StatefulEnvironment + ?Sized,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    A: Actor<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + ?Sized,
    H: SimulationHook<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + ?Sized,
    L: TimeSeriesLogger + ?Sized,
{
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
