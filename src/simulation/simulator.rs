/// Simulation trait and Simulator structs.
use crate::agents::{Agent, Step};
use crate::envs::StatefulEnvironment;
use crate::logging::{Event, Logger};
use crate::spaces::Space;

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
    A: Agent<
        <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
        <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
    >,
    L: Logger,
{
    environment: E,
    agent: A,
    logger: L,
}

impl<E, A, L> Simulator<E, A, L>
where
    E: StatefulEnvironment,
    A: Agent<
        <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
        <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
    >,
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
    A: Agent<
        <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
        <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
    >,
    L: Logger,
{
    fn run(&mut self, max_steps: Option<u64>) {
        run_with_logging(
            &mut self.environment,
            &mut self.agent,
            &mut self.logger,
            max_steps,
        );
    }
}

/// An simulator of a boxed agent and environment.
pub struct BoxedSimulator<OS: Space, AS: Space, L: Logger> {
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent: Box<dyn Agent<OS::Element, AS::Element>>,
    logger: L,
}

impl<OS: Space, AS: Space, L: Logger> BoxedSimulator<OS, AS, L> {
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

impl<OS: Space, AS: Space, L: Logger> Simulation for BoxedSimulator<OS, AS, L> {
    fn run(&mut self, max_steps: Option<u64>) {
        run_with_logging(
            self.environment.as_mut(),
            self.agent.as_mut(),
            &mut self.logger,
            max_steps,
        );
    }
}

/// Run an agent-environment simulation with logging.
pub fn run_with_logging<E, A, L>(
    environment: &mut E,
    agent: &mut A,
    logger: &mut L,
    max_steps: Option<u64>,
) where
    E: StatefulEnvironment + ?Sized,
    A: Agent<
            <<E as StatefulEnvironment>::ObservationSpace as Space>::Element,
            <<E as StatefulEnvironment>::ActionSpace as Space>::Element,
        > + ?Sized,
    L: Logger,
{
    let mut step_count = 0; // Global step count
    let mut episode_length = 0; // Length of the current episode in steps
    let mut episode_reward = 0.0; // Total reward for the current episode

    let structure = environment.structure();
    let observation_space = structure.observation_space;
    let action_space = structure.action_space;

    run(environment, agent, |step| {
        let reward = step.reward as f64;
        logger.log(Event::Step, "reward", reward.into()).unwrap();
        logger
            .log(
                Event::Step,
                "observation",
                observation_space.as_loggable(&step.observation),
            )
            .unwrap();
        logger
            .log(
                Event::Step,
                "action",
                action_space.as_loggable(&step.action),
            )
            .unwrap();
        logger.done(Event::Step);

        episode_length += 1;
        episode_reward += reward;
        if step.episode_done {
            logger
                .log(Event::Episode, "length", (episode_length as f64).into())
                .unwrap();
            episode_length = 0;
            logger
                .log(Event::Episode, "reward", episode_reward.into())
                .unwrap();
            episode_reward = 0.0;
            logger.done(Event::Episode);
        }

        step_count += 1;
        match max_steps {
            Some(steps) => step_count < steps,
            None => true,
        }
    });
}

/// Run an agent-environment simulation with a callback function called on each step.
///
/// The simulation will continue while the callback returns `true`.
pub fn run<E, A, F>(environment: &mut E, agent: &mut A, mut callback: F)
where
    E: StatefulEnvironment + ?Sized,
    A: Agent<
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
        let action = agent.act(&observation, new_episode);
        let (next_observation, reward, episode_done) = environment.step(&action);

        let step = Step {
            observation,
            action,
            reward,
            next_observation: next_observation.as_ref(),
            episode_done,
        };
        let stop = !callback(&step);
        agent.update(step);
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
