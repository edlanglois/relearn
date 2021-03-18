use crate::agents::{Agent, Step};
use crate::envs::StatefulEnvironment;
use crate::loggers::{Event, Logger};
use crate::spaces::Space;

/// An agent-environment simulator.
pub trait Simulator {
    /// Run the simulation for the specified number of steps.
    fn run(&mut self, max_steps: Option<u64>);
}

/// A simulator for a specific observation space, action space, and logger.
pub struct TypedSimulator<OS: Space, AS: Space, L: Logger> {
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent: Box<dyn Agent<OS::Element, AS::Element>>,
    logger: L,
}

impl<OS: Space, AS: Space, L: Logger> TypedSimulator<OS, AS, L> {
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

impl<OS: Space, AS: Space, L: Logger> Simulator for TypedSimulator<OS, AS, L> {
    fn run(&mut self, max_steps: Option<u64>) {
        let mut step_count = 0; // Global step count
        let mut episode_length = 0; // Length of the current episode in steps
        let mut episode_reward = 0.0; // Total reward for the current episode

        let structure = self.environment.structure();
        let observation_space = structure.observation_space;
        let action_space = structure.action_space;

        let logger = &mut self.logger;
        run(
            self.environment.as_mut(),
            self.agent.as_mut(),
            &mut |step| {
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
                    Some(steps) => step_count >= steps,
                    None => false,
                }
            },
        );
    }
}

/// Run an agent-environment simulation with a callback function called on each step.
pub fn run<OS, AS, F>(
    environment: &mut dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>,
    agent: &mut dyn Agent<OS::Element, AS::Element>,
    callback: &mut F,
) where
    OS: Space,
    AS: Space,
    F: FnMut(&Step<OS::Element, AS::Element>) -> bool,
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
        if callback(&step) {
            break;
        }
        agent.update(step);

        observation = if episode_done {
            environment.reset()
        } else {
            next_observation.expect("Observation must exist if the episode is not done")
        };
    }
}
