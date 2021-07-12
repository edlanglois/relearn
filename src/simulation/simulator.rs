use super::hooks::SimulationHook;
/// Simulation trait and Simulator structs.
use crate::agents::{Actor, Agent, Step};
use crate::envs::{EnvStructure, StatefulEnvironment};
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;

/// Runs a simulation.
pub trait Simulation {
    /// Run a simulation
    fn run(&mut self);
}

/// An agent-environment simulator with logging.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Simulator<E, A, L, H> {
    environment: E,
    agent: A,
    logger: L,
    hook: H,
}

impl<E, A, L, H> Simulator<E, A, L, H> {
    pub const fn new(environment: E, agent: A, logger: L, hook: H) -> Self {
        Self {
            environment,
            agent,
            logger,
            hook,
        }
    }
}

impl<E, A, L, H> Simulation for Simulator<E, A, L, H>
where
    E: StatefulEnvironment,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
        <<E as EnvStructure>::ObservationSpace as Space>::Element,
        <<E as EnvStructure>::ActionSpace as Space>::Element,
    >,
    L: TimeSeriesLogger,
    H: SimulationHook<
        <<E as EnvStructure>::ObservationSpace as Space>::Element,
        <<E as EnvStructure>::ActionSpace as Space>::Element,
        L,
    >,
{
    fn run(&mut self) {
        run_agent(
            &mut self.environment,
            &mut self.agent,
            &mut self.logger,
            &mut self.hook,
        );
    }
}

/// Run an agent-environment simulation.
///
/// # Args
/// * `environment` - The environment to simulate.
/// * `agent` - The agent to simulate.
/// * `logger` - The logger to use.
/// * `hook` - A simulation hook run on each step. Controls when the simulation stops.
pub fn run_agent<E, A, L, H>(environment: &mut E, agent: &mut A, logger: &mut L, hook: &mut H)
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
    // Not ?Sized because can't convert &(TimeSeriesLogger + ?Sized) => &mut dyn TimeSeriesLogger
    L: TimeSeriesLogger,
    H: SimulationHook<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
            L,
        > + ?Sized,
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
/// * `logger` - The logger to use. Passed to hook calls.
/// * `hook` - A simulation hook run on each step. Controls when the simulation stops.
pub fn run_actor<E, A, L, H>(environment: &mut E, actor: &mut A, logger: &mut L, hook: &mut H)
where
    E: StatefulEnvironment + ?Sized,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    A: Actor<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + ?Sized,
    L: TimeSeriesLogger + ?Sized,
    H: SimulationHook<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
            L,
        > + ?Sized,
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
