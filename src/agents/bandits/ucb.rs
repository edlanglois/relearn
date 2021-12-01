//! Upper confidence bound bandit agent.
use super::super::{
    ActorMode, BuildAgentError, BuildIndexAgent, FiniteSpaceAgent, PureActor, SetActorMode,
    SynchronousUpdate,
};
use crate::logging::TimeSeriesLogger;
use crate::simulation::TransientStep;
use crate::utils::iter::ArgMaxBy;
use ndarray::{Array, Array1, Array2, Axis};
use std::f64;
use std::fmt;

/// Configuration for a [`UCB1Agent`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UCB1AgentConfig {
    /// Scale factor on the confidence interval; controls the exploration rate.
    ///
    /// A value of 0.2 is recommended by Audibert and Munos in their ICML
    /// tutorial Introduction to Bandits: Algorithms and Theory (2011).
    pub exploration_rate: f64,
}

impl UCB1AgentConfig {
    pub const fn new(exploration_rate: f64) -> Self {
        Self { exploration_rate }
    }
}

impl Default for UCB1AgentConfig {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl BuildIndexAgent for UCB1AgentConfig {
    type Agent = BaseUCB1Agent;

    fn build_index_agent(
        &self,
        num_observations: usize,
        num_actions: usize,
        reward_range: (f64, f64),
        _discount_factor: f64,
        _seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        BaseUCB1Agent::new(
            num_observations,
            num_actions,
            reward_range,
            self.exploration_rate,
        )
    }
}

/// A UCB1 Agent
///
/// Applies UCB1 (Auer 2002) independently to each state.
pub type UCB1Agent<OS, AS> = FiniteSpaceAgent<BaseUCB1Agent, OS, AS>;

/// Base UCB1 Agent
///
/// Applies UCB1 (Auer 2002) independently to each state.
/// Defined for index observation and action spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct BaseUCB1Agent {
    /// Scale factor on the confidence interval; controls the exploration rate.
    ///
    /// A value of 0.2 is recommended by Audibert and Munos in their ICML
    /// tutorial Introduction to Bandits: Algorithms and Theory (2011).
    pub exploration_rate: f64,

    /// Mode of actor behaviour
    pub mode: ActorMode,

    // Parameters to scale the reward to [0, 1]
    reward_scale_factor: f64,
    reward_shift: f64,

    /// The mean reward for each state-action pair
    state_action_mean_reward: Array2<f64>,
    /// The selection count for each state-action pair
    state_action_count: Array2<u64>,
    /// The visit count for each state
    state_visit_count: Array1<u64>,
}

impl BaseUCB1Agent {
    pub fn new(
        num_observations: usize,
        num_actions: usize,
        reward_range: (f64, f64),
        exploration_rate: f64,
    ) -> Result<Self, BuildAgentError> {
        let (min_reward, max_reward) = reward_range;

        let reward_width = max_reward - min_reward;
        if !reward_width.is_finite() {
            return Err(BuildAgentError::UnboundedReward);
        }
        let reward_scale_factor = reward_width.recip();
        let reward_shift = -min_reward;

        // Initialize to 1 success and 1 failure for each arm
        let state_action_mean_reward = Array::from_elem((num_observations, num_actions), 0.5);
        let state_action_count = Array::from_elem((num_observations, num_actions), 2);
        let state_visit_count = Array::from_elem((num_observations,), 2 * num_actions as u64);

        Ok(Self {
            exploration_rate,
            mode: ActorMode::Training,
            reward_scale_factor,
            reward_shift,
            state_action_mean_reward,
            state_action_count,
            state_visit_count,
        })
    }
}

impl fmt::Display for BaseUCB1Agent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BaseUCB1Agent(ϵ={})", self.exploration_rate)
    }
}

impl BaseUCB1Agent {
    /// Take a training-mode action.
    fn act_training(&self, obs_idx: usize) -> usize {
        let log_squared_visit_count = 2.0 * (self.state_visit_count[obs_idx] as f64).ln();
        let ucb = self
            .state_action_count
            .index_axis(Axis(0), obs_idx)
            .mapv(|action_count| {
                (log_squared_visit_count / (action_count as f64)).sqrt() * self.exploration_rate
            })
            + self.state_action_mean_reward.index_axis(Axis(0), obs_idx);
        ucb.into_iter()
            .argmax_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Empty action space")
    }

    /// Take a release-mode (greedy) action.
    fn act_release(&self, obs_idx: usize) -> usize {
        // Take the action with the largest action count
        self.state_action_count
            .index_axis(Axis(0), obs_idx)
            .into_iter()
            .argmax_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Empty action space")
    }
}

impl PureActor<usize, usize> for BaseUCB1Agent {
    type State = ();

    #[inline]
    fn initial_state(&self, _seed: u64) -> Self::State {}

    #[inline]
    fn reset_state(&self, _state: &mut Self::State) {}

    #[inline]
    fn act(&self, _: &mut Self::State, observation: &usize) -> usize {
        match self.mode {
            ActorMode::Training => self.act_training(*observation),
            ActorMode::Release => self.act_release(*observation),
        }
    }
}

impl SynchronousUpdate<usize, usize> for BaseUCB1Agent {
    fn update(&mut self, step: TransientStep<usize, usize>, _logger: &mut dyn TimeSeriesLogger) {
        let scaled_reward = (step.reward + self.reward_shift) * self.reward_scale_factor;

        self.state_visit_count[step.observation] += 1;
        let state_action_count = self
            .state_action_count
            .get_mut((step.observation, step.action))
            .unwrap();
        *state_action_count += 1;
        let mean_reward = self
            .state_action_mean_reward
            .get_mut((step.observation, step.action))
            .unwrap();
        *mean_reward += (scaled_reward - *mean_reward) / (*state_action_count as f64);
    }
}

impl SetActorMode for BaseUCB1Agent {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.mode = mode
    }
}

#[cfg(test)]
mod ucb1_agent {
    use super::super::super::{testing, BuildAgent};
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        let config = UCB1AgentConfig::default();
        testing::pure_train_deterministic_bandit(
            |env| config.build_agent(env, 0).unwrap(),
            1000,
            0.9,
        );
    }
}
