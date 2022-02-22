use super::PartialStep;
use std::fmt;
use std::iter::FromIterator;

/// Basic summary statistics of simulation steps.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct StepsSummary {
    pub num_steps: u64,
    pub num_episodes: u64,
    pub total_reward: f64,
}

impl fmt::Display for StepsSummary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "num_steps: {}", self.num_steps)?;
        writeln!(f, "num_episodes: {}", self.num_episodes)?;
        writeln!(
            f,
            "mean_step_reward: {}",
            self.total_reward / self.num_steps as f64
        )?;
        writeln!(
            f,
            "mean_ep_reward:   {}",
            self.total_reward / self.num_episodes as f64
        )?;
        writeln!(
            f,
            "mean_ep_length:   {}",
            self.num_steps as f64 / self.num_episodes as f64
        )?;
        Ok(())
    }
}

impl StepsSummary {
    pub fn update<O, A>(&mut self, step: &PartialStep<O, A>) {
        self.num_steps += 1;
        self.total_reward += step.reward;
        if step.next.episode_done() {
            self.num_episodes += 1;
        }
    }
}

impl<O, A> FromIterator<PartialStep<O, A>> for StepsSummary {
    fn from_iter<I>(steps: I) -> Self
    where
        I: IntoIterator<Item = PartialStep<O, A>>,
    {
        steps.into_iter().fold(Self::default(), |mut s, step| {
            s.update(&step);
            s
        })
    }
}
