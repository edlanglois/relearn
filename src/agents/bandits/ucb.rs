//! Upper confidence bound bandit agent.
use super::super::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::spaces::FiniteSpace;
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

impl<E> AgentBuilder<UCB1Agent<E::ObservationSpace, E::ActionSpace>, E> for UCB1AgentConfig
where
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: FiniteSpace,
    <E as EnvStructure>::ActionSpace: FiniteSpace,
{
    fn build_agent(
        &self,
        env: &E,
        _seed: u64,
    ) -> Result<UCB1Agent<E::ObservationSpace, E::ActionSpace>, BuildAgentError> {
        UCB1Agent::new(
            env.observation_space(),
            env.action_space(),
            env.reward_range(),
            self.exploration_rate,
        )
    }
}

/// A UCB1 Agent
///
/// Applies UCB1 (Auer 2002) independently to each state.
#[derive(Debug, Clone, PartialEq)]
pub struct UCB1Agent<OS, AS> {
    /// Environment observation space
    pub observation_space: OS,
    /// Environment action space
    pub action_space: AS,

    /// Scale factor on the confidence interval; controls the exploration rate.
    ///
    /// A value of 0.2 is recommended by Audibert and Munos in their ICML
    /// tutorial Introduction to Bandits: Algorithms and Theory (2011).
    pub exploration_rate: f64,

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

impl<OS, AS> UCB1Agent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    pub fn new(
        observation_space: OS,
        action_space: AS,
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

        let num_observations = observation_space.size();
        let num_actions = action_space.size();

        // Initialize to 1 success and 1 failure for each arm
        let state_action_mean_reward = Array::from_elem((num_observations, num_actions), 0.5);
        let state_action_count = Array::from_elem((num_observations, num_actions), 2);
        let state_visit_count = Array::from_elem((num_observations,), 2 * num_actions as u64);

        Ok(Self {
            observation_space,
            action_space,
            exploration_rate,
            reward_scale_factor,
            reward_shift,
            state_action_mean_reward,
            state_action_count,
            state_visit_count,
        })
    }
}

impl<OS, AS> fmt::Display for UCB1Agent<OS, AS>
where
    OS: fmt::Display,
    AS: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "UCB1Agent({}, {}, {})",
            self.observation_space, self.action_space, self.exploration_rate
        )
    }
}

impl<OS, AS> Actor<OS::Element, AS::Element> for UCB1Agent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn act(&mut self, observation: &OS::Element, _new_episode: bool) -> AS::Element {
        let obs_idx = self.observation_space.to_index(observation);
        // Take the action with the largest action count
        let act_idx = self
            .state_action_count
            .index_axis(Axis(0), obs_idx)
            .into_iter()
            .argmax_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Empty action space");
        self.action_space.from_index(act_idx).unwrap()
    }
}

impl<OS, AS> Agent<OS::Element, AS::Element> for UCB1Agent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn act(&mut self, observation: &OS::Element, _new_episode: bool) -> AS::Element {
        let obs_idx = self.observation_space.to_index(observation);
        let log_squared_visit_count = 2.0 * (self.state_visit_count[obs_idx] as f64).ln();
        let ucb = self
            .state_action_count
            .index_axis(Axis(0), obs_idx)
            .mapv(|action_count| {
                (log_squared_visit_count / (action_count as f64)).sqrt() * self.exploration_rate
            })
            + self.state_action_mean_reward.index_axis(Axis(0), obs_idx);
        let act_idx = ucb
            .into_iter()
            .argmax_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Empty action space");
        self.action_space.from_index(act_idx).unwrap()
    }
    fn update(&mut self, step: Step<OS::Element, AS::Element>, _logger: &mut dyn TimeSeriesLogger) {
        let obs_idx = self.observation_space.to_index(&step.observation);
        let act_idx = self.action_space.to_index(&step.action);

        let scaled_reward = (step.reward + self.reward_shift) * self.reward_scale_factor;

        self.state_visit_count[obs_idx] += 1;
        let state_action_count = self.state_action_count.get_mut((obs_idx, act_idx)).unwrap();
        *state_action_count += 1;
        let mean_reward = self
            .state_action_mean_reward
            .get_mut((obs_idx, act_idx))
            .unwrap();
        *mean_reward += (scaled_reward - *mean_reward) / (*state_action_count as f64);
    }
}

#[cfg(test)]
mod ucb1_agent {
    use super::super::super::testing;
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        let config = UCB1AgentConfig::default();
        testing::train_deterministic_bandit(
            |env_structure| config.build_agent(env_structure, 0).unwrap(),
            1000,
            0.9,
        );
    }
}
