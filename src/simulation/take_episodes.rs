use super::{PartialStep, Simulation};
use std::iter::FusedIterator;

/// An iterator that iterates over the first `n` episodes of `steps`.
#[derive(Debug, Default, Clone)]
pub struct TakeEpisodes<I> {
    steps: I,
    n: usize,
}

impl<I> TakeEpisodes<I> {
    #[inline]
    pub const fn new(steps: I, n: usize) -> Self {
        Self { steps, n }
    }
}

impl<I> Simulation for TakeEpisodes<I>
where
    I: Simulation,
{
    type Observation = I::Observation;
    type Action = I::Action;
    type Environment = I::Environment;
    type Actor = I::Actor;
    type Logger = I::Logger;

    #[inline]
    fn env(&self) -> &Self::Environment {
        self.steps.env()
    }
    #[inline]
    fn env_mut(&mut self) -> &mut Self::Environment {
        self.steps.env_mut()
    }
    #[inline]
    fn actor(&self) -> &Self::Actor {
        self.steps.actor()
    }
    #[inline]
    fn actor_mut(&mut self) -> &mut Self::Actor {
        self.steps.actor_mut()
    }
    #[inline]
    fn logger(&self) -> &Self::Logger {
        self.steps.logger()
    }
    #[inline]
    fn logger_mut(&mut self) -> &mut Self::Logger {
        self.steps.logger_mut()
    }
}

impl<I, O, A> Iterator for TakeEpisodes<I>
where
    I: Iterator<Item = PartialStep<O, A>>,
{
    type Item = PartialStep<O, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.n > 0 {
            let step = self.steps.next();
            if let Some(s) = &step {
                if s.episode_done() {
                    self.n -= 1;
                }
            }
            step
        } else {
            None
        }
    }
}

impl<I, O, A> FusedIterator for TakeEpisodes<I> where I: FusedIterator<Item = PartialStep<O, A>> {}

#[cfg(test)]
mod tests {
    use crate::agents::RandomAgent;
    use crate::envs::Chain;
    use crate::envs::{EnvStructure, Environment};
    use crate::simulation::{SimSeed, StepsIter, StepsSummary};

    #[allow(clippy::cast_possible_truncation)]
    #[test]
    fn episode_count() {
        let steps_per_episode = 10;
        let num_episodes = 30;

        let env = Chain::default().with_latent_step_limit(steps_per_episode);
        let agent = RandomAgent::new(env.action_space());
        let summary: StepsSummary = env
            .run(agent, SimSeed::Root(53), ())
            // Additional step bound so that the test does not hang if take_episodes breaks
            .take((5 * steps_per_episode * num_episodes) as usize)
            .take_episodes(num_episodes as usize)
            .collect();
        assert_eq!(summary.num_episodes(), num_episodes);
        assert_eq!(summary.num_steps(), steps_per_episode * num_episodes);
    }
}
