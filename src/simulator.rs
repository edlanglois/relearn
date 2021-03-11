use crate::agents::{Agent, Step};
use crate::envs::Environment;

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
