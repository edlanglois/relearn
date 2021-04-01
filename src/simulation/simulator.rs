/// Simulation trait and Simulator structs.
use crate::agents::{Actor, Agent, Step};
use crate::envs::{EnvStructure, StatefulEnvironment};
use crate::logging::{Event, Loggable, Logger, NullLogger};
use crate::spaces::{ElementRefInto, Space};

/// Runs a simulation.
pub trait Simulation {
    /// Run a simulation for the specified number of steps.
    fn run(&mut self, max_steps: Option<u64>);
}

/// An agent-environment simulator with logging.
#[derive(Debug)]
pub struct Simulator<E, A, L>
where
    E: StatefulEnvironment,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
        <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
        <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
    >,
    <E as StatefulEnvironment>::ObservationSpace: ElementRefInto<Loggable>,
    <E as StatefulEnvironment>::ActionSpace: ElementRefInto<Loggable>,
    L: Logger,
{
    environment: E,
    agent: A,
    logger: L,
}

impl<E, A, L> Simulator<E, A, L>
where
    E: StatefulEnvironment,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
        <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
        <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
    >,
    <E as StatefulEnvironment>::ObservationSpace: ElementRefInto<Loggable>,
    <E as StatefulEnvironment>::ActionSpace: ElementRefInto<Loggable>,
    L: Logger,
{
    pub fn new(environment: E, agent: A, logger: L) -> Self {
        Self {
            environment,
            agent,
            logger,
        }
    }
}

impl<E, A, L> Simulation for Simulator<E, A, L>
where
    E: StatefulEnvironment,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
        <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
        <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
    >,
    <E as StatefulEnvironment>::ObservationSpace: ElementRefInto<Loggable>,
    <E as StatefulEnvironment>::ActionSpace: ElementRefInto<Loggable>,
    L: Logger,
{
    fn run(&mut self, max_steps: Option<u64>) {
        run_with_logging(
            &mut self.environment,
            &mut self.agent,
            &mut self.logger,
            max_steps,
            |_| (),
        );
    }
}

/// An simulator of a boxed agent and environment.
pub struct BoxedSimulator<OS: Space, AS: Space, L: Logger> {
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent: Box<dyn Agent<OS::Element, AS::Element>>,
    logger: L,
}

impl<OS, AS, L> BoxedSimulator<OS, AS, L>
where
    OS: Space,
    <OS as Space>::Element: Clone,
    AS: Space,
    L: Logger,
{
    pub fn new(
        environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
        agent: Box<dyn Agent<OS::Element, AS::Element>>,
        logger: L,
    ) -> Self {
        Self {
            environment,
            agent,
            logger,
        }
    }
}

impl<OS, AS, L> Simulation for BoxedSimulator<OS, AS, L>
where
    OS: Space + ElementRefInto<Loggable>,
    <OS as Space>::Element: Clone,
    AS: Space + ElementRefInto<Loggable>,
    L: Logger,
{
    fn run(&mut self, max_steps: Option<u64>) {
        run_with_logging(
            self.environment.as_mut(),
            self.agent.as_mut(),
            &mut self.logger,
            max_steps,
            |_| (),
        );
    }
}

/// Run an agent-environment simulation with logging.
pub fn run_with_logging<E, A, L, F>(
    environment: &mut E,
    agent: &mut A,
    logger: &mut L,
    max_steps: Option<u64>,
    mut callback: F,
) where
    E: StatefulEnvironment + ?Sized,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        > + ?Sized,
    <E as StatefulEnvironment>::ObservationSpace: ElementRefInto<Loggable>,
    <E as StatefulEnvironment>::ActionSpace: ElementRefInto<Loggable>,
    L: Logger,
    F: FnMut(
        &Step<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        >,
    ),
{
    let mut step_count = 0; // Global step count
    let mut tracker = SimulationTracker::new(environment.structure());

    run_(environment, agent, logger, |step, logger| {
        tracker.log_step(step, logger);
        callback(step);

        step_count += 1;
        match max_steps {
            Some(steps) => step_count < steps,
            None => true,
        }
    });
}

/// Run an agent-environment simulation.
pub fn run<E, A, F>(environment: &mut E, agent: &mut A, max_steps: Option<u64>, mut callback: F)
where
    E: StatefulEnvironment + ?Sized,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        > + ?Sized,
    F: FnMut(
        &Step<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        >,
    ),
{
    let mut step_count = 0; // Global step count
    let mut logger = NullLogger::new();

    run_(environment, agent, &mut logger, |step, _| {
        callback(step);

        step_count += 1;
        match max_steps {
            Some(steps) => step_count < steps,
            None => true,
        }
    });
}

/// Run an agent-environment simulation with a callback and logger pass-through to the agent.
fn run_<E, A, F, L>(environment: &mut E, agent: &mut A, logger: &mut L, mut callback: F)
where
    E: StatefulEnvironment + ?Sized,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Clone,
    A: Agent<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        > + ?Sized,
    F: FnMut(
        &Step<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        >,
        &mut L,
    ) -> bool,
    L: Logger,
{
    let mut observation = environment.reset();
    let new_episode = true;

    loop {
        let action = agent.act(&observation, new_episode);
        let (next_observation, reward, episode_done) = environment.step(&action);

        let step = Step {
            observation,
            action,
            reward,
            next_observation: next_observation.clone(),
            episode_done,
        };
        let stop = !callback(&step, logger);
        agent.update(step, logger);
        if stop {
            break;
        }

        observation = if episode_done {
            environment.reset()
        } else {
            next_observation.expect("Observation must exist if the episode is not done")
        };
    }
}

/// Run an actor-environment simulation with a callback function called on each step.
///
/// The simulation will continue while the callback returns `true`.
///
/// An actor never learns between episodes.
/// An actor may depend on history from the current episode.
pub fn run_actor<E, A, F>(environment: &mut E, actor: &mut A, mut callback: F)
where
    E: StatefulEnvironment + ?Sized,
    <<E as StatefulEnvironment>::ObservationSpace as Space>::Element: Clone,
    A: Actor<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        > + ?Sized,
    F: FnMut(
        &Step<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        >,
    ) -> bool,
{
    let mut observation = environment.reset();
    let new_episode = true;

    loop {
        let action = actor.act(&observation, new_episode);
        let (next_observation, reward, episode_done) = environment.step(&action);

        let step = Step {
            observation,
            action,
            reward,
            next_observation: next_observation.clone(),
            episode_done,
        };
        if !callback(&step) {
            break;
        }

        observation = if episode_done {
            environment.reset()
        } else {
            next_observation.expect("Observation must exist if the episode is not done")
        };
    }
}

/// Track simulation progress for logging
struct SimulationTracker<OS, AS>
where
    OS: ElementRefInto<Loggable>,
    AS: ElementRefInto<Loggable>,
{
    pub observation_space: OS,
    pub action_space: AS,

    pub episode_length: u64,
    pub episode_reward: f64,
}

impl<OS, AS> SimulationTracker<OS, AS>
where
    OS: ElementRefInto<Loggable>,
    AS: ElementRefInto<Loggable>,
{
    pub fn new(env_structure: EnvStructure<OS, AS>) -> Self {
        Self {
            observation_space: env_structure.observation_space,
            action_space: env_structure.action_space,
            episode_length: 0,
            episode_reward: 0.0,
        }
    }

    pub fn log_step<L: Logger>(&mut self, step: &Step<OS::Element, AS::Element>, logger: &mut L) {
        let reward = step.reward as f64;
        logger.log(Event::Step, "reward", reward.into()).unwrap();
        logger
            .log(
                Event::Step,
                "observation",
                self.observation_space.elem_ref_into(&step.observation),
            )
            .unwrap();
        logger
            .log(
                Event::Step,
                "action",
                self.action_space.elem_ref_into(&step.action),
            )
            .unwrap();
        logger.done(Event::Step);

        self.episode_length += 1;
        self.episode_reward += reward;
        if step.episode_done {
            logger
                .log(
                    Event::Episode,
                    "length",
                    (self.episode_length as f64).into(),
                )
                .unwrap();
            self.episode_length = 0;
            logger
                .log(Event::Episode, "reward", self.episode_reward.into())
                .unwrap();
            self.episode_reward = 0.0;
            logger.done(Event::Episode);
        }
    }
}
