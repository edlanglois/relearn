//! Thompson sampling bandit agent
use super::super::{
    buffers::SimpleBuffer, finite::FiniteSpaceAgent, Actor, ActorMode, Agent, BatchUpdate,
    BufferCapacityBound, BuildAgent, BuildAgentError,
};
use crate::envs::EnvStructure;
use crate::logging::StatsLogger;
use crate::simulation::PartialStep;
use crate::spaces::FiniteSpace;
use crate::utils::iter::ArgMaxBy;
use crate::Prng;
use ndarray::{Array, Array2, Axis};
use rand::distributions::Distribution;
use rand_distr::Beta;
use std::fmt;
use std::sync::Arc;

/// Configuration for [`BetaThompsonSamplingAgent`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BetaThompsonSamplingAgentConfig {
    /// Number of posterior samples to draw.
    /// Takes the action with the highest mean sampled value.
    pub num_samples: usize,
}

impl BetaThompsonSamplingAgentConfig {
    pub const fn new(num_samples: usize) -> Self {
        Self { num_samples }
    }
}

impl Default for BetaThompsonSamplingAgentConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<OS, AS> BuildAgent<OS, AS> for BetaThompsonSamplingAgentConfig
where
    OS: FiniteSpace + Clone + 'static,
    AS: FiniteSpace + Clone + 'static,
{
    type Agent = BetaThompsonSamplingAgent<OS, AS>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        let observation_space = env.observation_space();
        let action_space = env.action_space();
        Ok(FiniteSpaceAgent {
            agent: BaseBetaThompsonSamplingAgent::new(
                observation_space.size(),
                action_space.size(),
                env.reward_range(),
                self.num_samples,
            ),
            observation_space,
            action_space,
        })
    }
}

/// A Thompson sampling agent for Bernoulli rewards with a Beta prior.
pub type BetaThompsonSamplingAgent<OS, AS> =
    FiniteSpaceAgent<BaseBetaThompsonSamplingAgent, OS, AS>;

/// Base Thompson sampling agent for Bernoulli rewards with a Beta prior.
///
/// Implemented only for index action and observation spaces.
#[derive(Debug, PartialEq)]
pub struct BaseBetaThompsonSamplingAgent {
    /// Reward is partitioned into high/low separated by this threshold.
    pub reward_threshold: f64,
    /// Number of posterior samples to draw.
    /// Takes the action with the highest mean sampled value.
    pub num_samples: usize,

    /// Count of low and high rewards for each observation-action pair.
    low_high_reward_counts: Arc<Array2<(u64, u64)>>,
}

impl BaseBetaThompsonSamplingAgent {
    pub fn new(
        num_observations: usize,
        num_actions: usize,
        reward_range: (f64, f64),
        num_samples: usize,
    ) -> Self {
        let (reward_min, reward_max) = reward_range;
        let reward_threshold = (reward_min + reward_max) / 2.0;
        let low_high_reward_counts =
            Arc::new(Array::from_elem((num_observations, num_actions), (1, 1)));
        Self {
            reward_threshold,
            num_samples,
            low_high_reward_counts,
        }
    }
}

impl fmt::Display for BaseBetaThompsonSamplingAgent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BaseBetaThompsonSamplingAgent({})",
            self.reward_threshold
        )
    }
}

impl BaseBetaThompsonSamplingAgent {
    /// Update based on an on-policy or off-policy step.
    fn step_update(&mut self, step: PartialStep<usize, usize>) {
        let reward_count = Arc::get_mut(&mut self.low_high_reward_counts)
            .expect("cannot update agent while actors exist")
            .get_mut((step.observation, step.action))
            .unwrap();
        if step.reward > self.reward_threshold {
            reward_count.1 += 1;
        } else {
            reward_count.0 += 1;
        }
    }
}

impl Agent<usize, usize> for BaseBetaThompsonSamplingAgent {
    type Actor = BaseBetaThompsonSamplingActor;

    fn actor(&self, mode: ActorMode) -> Self::Actor {
        BaseBetaThompsonSamplingActor {
            mode,
            num_samples: self.num_samples,
            low_high_reward_counts: Arc::clone(&self.low_high_reward_counts),
        }
    }
}

impl BatchUpdate<usize, usize> for BaseBetaThompsonSamplingAgent {
    type HistoryBuffer = SimpleBuffer<usize, usize>;

    fn batch_size_hint(&self) -> BufferCapacityBound {
        BufferCapacityBound {
            min_steps: 1,
            min_incomplete_episode_len: Some(0),
            ..BufferCapacityBound::default()
        }
    }

    fn buffer(&self, capacity: BufferCapacityBound) -> Self::HistoryBuffer {
        SimpleBuffer::new(capacity)
    }

    fn batch_update<'a, I>(&mut self, buffers: I, _logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        for buffer in buffers {
            for step in buffer.drain_steps() {
                self.step_update(step)
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct BaseBetaThompsonSamplingActor {
    mode: ActorMode,
    num_samples: usize,
    low_high_reward_counts: Arc<Array2<(u64, u64)>>,
}

impl Actor<usize, usize> for BaseBetaThompsonSamplingActor {
    type EpisodeState = ();

    fn new_episode_state(&self, _: &mut Prng) -> Self::EpisodeState {}

    fn act(&self, _: &mut Self::EpisodeState, observation: &usize, rng: &mut Prng) -> usize {
        match self.mode {
            ActorMode::Training => self
                .low_high_reward_counts
                .index_axis(Axis(0), *observation)
                .mapv(|(beta, alpha)| -> f64 {
                    Beta::new(alpha as f64, beta as f64)
                        .unwrap()
                        .sample_iter(&mut *rng)
                        .take(self.num_samples)
                        .sum()
                })
                .into_iter()
                .argmax_by(|a, b| a.partial_cmp(b).unwrap())
                .expect("empty action space"),
            ActorMode::Evaluation => self
                .low_high_reward_counts
                .index_axis(Axis(0), *observation)
                .mapv(|(beta, alpha)| alpha as f64 / (alpha + beta) as f64)
                .into_iter()
                .argmax_by(|a, b| a.partial_cmp(b).unwrap())
                .expect("empty action space"),
        }
    }
}

#[cfg(test)]
mod beta_thompson_sampling {
    use super::super::super::testing;
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        testing::train_deterministic_bandit(&BetaThompsonSamplingAgentConfig::default(), 1000, 0.9);
    }
}
