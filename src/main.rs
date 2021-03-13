use rust_rl::agents::TabularQLearningAgent;
use rust_rl::envs::{BernoulliBandit, EnvSpec};
use rust_rl::loggers::{CLILogger, Event, Logger};
use rust_rl::simulator;
use std::time::Duration;

fn main() {
    let seed = 1;
    let mut environment = BernoulliBandit::new(vec![0.2, 0.8], seed);
    println!("Environment: {}", environment);
    // let mut agent = RandomAgent::new(environment.action_space(), seed + 1);
    let mut agent = TabularQLearningAgent::new(
        environment.observation_space(),
        environment.action_space(),
        1.0,
        0.2,
        seed + 1,
    );
    println!("Agent: {}", agent);

    let num_steps = 10_000_000;
    let mut logger = CLILogger::new(Duration::from_millis(1000), true);

    let mut step_count = 0;
    let mut episode_length = 0;
    let mut episode_reward = 0.0;
    simulator::run(&mut environment, &mut agent, &mut |step| {
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
        step_count >= num_steps
    });
}
