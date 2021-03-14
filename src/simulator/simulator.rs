use crate::agents::{Agent, Step};
use crate::envs::Environment;
use crate::loggers::{Event, Logger};

/// An agent-environment simulator.
pub trait Simulator {
    /// Run the simulation for the specified number of steps.
    fn run(&mut self, max_steps: Option<u64>);
}

/// A simulator templated on the observation and action spaces, as well as the logger.
pub struct TypedSimulator<O, A, L: Logger> {
    environment: Box<dyn Environment<Observation = O, Action = A>>,
    agent: Box<dyn Agent<O, A>>,
    logger: L,
}

impl<O, A, L: Logger> TypedSimulator<O, A, L> {
    pub fn new(
        environment: Box<dyn Environment<Observation = O, Action = A>>,
        agent: Box<dyn Agent<O, A>>,
        logger: L,
    ) -> Self {
        Self {
            environment,
            agent,
            logger,
        }
    }
}

impl<O, A, L: Logger> Simulator for TypedSimulator<O, A, L> {
    fn run(&mut self, max_steps: Option<u64>) {
        let mut step_count = 0; // Global step count
        let mut episode_length = 0; // Length of the current episode in steps
        let mut episode_reward = 0.0; // Total reward for the current episode

        let logger = &mut self.logger;
        run(
            self.environment.as_mut(),
            self.agent.as_mut(),
            &mut |step| {
                let reward = step.reward as f64;
                logger.log_scalar(Event::Step, "reward", reward);
                logger.done(Event::Step);

                episode_length += 1;
                episode_reward += reward;
                if step.episode_done {
                    logger.log_scalar(Event::Episode, "length", episode_length as f64);
                    episode_length = 0;
                    logger.log_scalar(Event::Episode, "reward", episode_reward);
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
pub fn run<O, A, F>(
    environment: &mut dyn Environment<Observation = O, Action = A>,
    agent: &mut dyn Agent<O, A>,
    callback: &mut F,
) where
    F: FnMut(&Step<O, A>) -> bool,
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
