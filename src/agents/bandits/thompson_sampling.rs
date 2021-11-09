//! Thompson sampling bandit agent
use super::super::{
    Actor, ActorMode, Agent, BuildAgentError, BuildIndexAgent, FiniteSpaceAgent, OffPolicyAgent,
    SetActorMode, Step, SyncParams, SyncParamsError,
};
use crate::logging::TimeSeriesLogger;
use crate::utils::iter::ArgMaxBy;
use ndarray::{Array, Array2, Axis};
use rand::distributions::Distribution;
use rand::prelude::*;
use rand_distr::Beta;
use std::fmt;

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

impl BuildIndexAgent for BetaThompsonSamplingAgentConfig {
    type Agent = BaseBetaThompsonSamplingAgent;

    fn build_index_agent(
        &self,
        num_observations: usize,
        num_actions: usize,
        reward_range: (f64, f64),
        _discount_factor: f64,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(BaseBetaThompsonSamplingAgent::new(
            num_observations,
            num_actions,
            reward_range,
            self.num_samples,
            seed,
        ))
    }
}

/// A Thompson sampling agent for Bernoulli rewards with a Beta prior.
pub type BetaThompsonSamplingAgent<OS, AS> =
    FiniteSpaceAgent<BaseBetaThompsonSamplingAgent, OS, AS>;

/// Base Thompson sampling agent for Bernoulli rewards with a Beta prior.
///
/// Implemented only for index action and observation spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct BaseBetaThompsonSamplingAgent {
    /// Reward is partitioned into high/low separated by this threshold.
    pub reward_threshold: f64,
    /// Number of posterior samples to draw.
    /// Takes the action with the highest mean sampled value.
    pub num_samples: usize,
    /// Mode of actor behaviour
    pub mode: ActorMode,

    /// Count of low and high rewards for each observation-action pair.
    low_high_reward_counts: Array2<(u64, u64)>,

    rng: StdRng,
}

impl BaseBetaThompsonSamplingAgent {
    pub fn new(
        num_observations: usize,
        num_actions: usize,
        reward_range: (f64, f64),
        num_samples: usize,
        seed: u64,
    ) -> Self {
        let (reward_min, reward_max) = reward_range;
        let reward_threshold = (reward_min + reward_max) / 2.0;
        let low_high_reward_counts = Array::from_elem((num_observations, num_actions), (1, 1));
        Self {
            reward_threshold,
            num_samples,
            mode: ActorMode::Training,
            low_high_reward_counts,
            rng: StdRng::seed_from_u64(seed),
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
    /// Take a training-mode action.
    fn act_training(&mut self, obs_idx: usize) -> usize {
        let num_samples = self.num_samples;
        let rng = &mut self.rng;
        self.low_high_reward_counts
            .index_axis(Axis(0), obs_idx)
            .mapv(|(beta, alpha)| -> f64 {
                // Explanation for the rng reference:
                // sample_iter takes its argument by value rather than by reference.
                // We cannot the rng into sample_iter because it needs to stay with self.
                // However, (&mut rng) also implements Rng.
                // We cannot directly use &mut self.rng in the closure because that would borrow
                // self. Nor can we create a copy like we do for num_samples.
                // So we create `rng` as a reference.
                // We cannot directly pass in `rng` because that would move it out of `rng` and
                // it needs to be available for multiple function calls.
                //
                // Therefore, the solution is to dereference rng first so that we have back the
                // original rng (without naming `self` within the closure) and then reference it
                // again in the closure so that the reference is created local to the closure and
                // can be safely moved.
                Beta::new(alpha as f64, beta as f64)
                    .unwrap()
                    .sample_iter(&mut *rng)
                    .take(num_samples)
                    .sum()
            })
            .into_iter()
            .argmax_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Empty action space")
    }

    /// Take a release-mode (greedy) action.
    fn act_release(&mut self, obs_idx: usize) -> usize {
        // Take the action with highest posterior mean
        // Counts are initalized to 1 so there is no risk of 0/0
        self.low_high_reward_counts
            .index_axis(Axis(0), obs_idx)
            .mapv(|(beta, alpha)| alpha as f64 / (alpha + beta) as f64)
            .into_iter()
            .argmax_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Empty action space")
    }
}

impl Actor<usize, usize> for BaseBetaThompsonSamplingAgent {
    fn act(&mut self, observation: &usize, _new_episode: bool) -> usize {
        match self.mode {
            ActorMode::Training => self.act_training(*observation),
            ActorMode::Release => self.act_release(*observation),
        }
    }
}

impl Agent<usize, usize> for BaseBetaThompsonSamplingAgent {
    fn update(&mut self, step: Step<usize, usize>, _logger: &mut dyn TimeSeriesLogger) {
        let reward_count = self
            .low_high_reward_counts
            .get_mut((step.observation, step.action))
            .unwrap();
        if step.reward > self.reward_threshold {
            reward_count.1 += 1;
        } else {
            reward_count.0 += 1;
        }
    }
}

impl OffPolicyAgent for BaseBetaThompsonSamplingAgent {}

impl SetActorMode for BaseBetaThompsonSamplingAgent {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.mode = mode
    }
}

impl SyncParams for BaseBetaThompsonSamplingAgent {
    fn sync_params(&mut self, target: &Self) -> Result<(), SyncParamsError> {
        if self.low_high_reward_counts.raw_dim() == target.low_high_reward_counts.raw_dim() {
            self.low_high_reward_counts
                .assign(&target.low_high_reward_counts);
            Ok(())
        } else {
            Err(SyncParamsError::IncompatibleParams)
        }
    }
}

#[cfg(test)]
mod beta_thompson_sampling {
    use super::super::super::{testing, BuildAgent};
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        let config = BetaThompsonSamplingAgentConfig::default();
        testing::train_deterministic_bandit(|env| config.build_agent(env, 0).unwrap(), 1000, 0.9);
    }
}
