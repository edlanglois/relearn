//! Upper confidence bound bandit agent.
use super::super::{
    buffers::VecBuffer, finite::FiniteSpaceAgent, Actor, ActorMode, Agent, BatchUpdate, BuildAgent,
    BuildAgentError, HistoryDataBound,
};
use crate::envs::EnvStructure;
use crate::feedback::Reward;
use crate::logging::StatsLogger;
use crate::simulation::PartialStep;
use crate::spaces::{FiniteSpace, IntervalSpace};
use crate::utils::iter::ArgMaxBy;
use crate::Prng;
use ndarray::{Array, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::f64;
use std::fmt;
use std::sync::Arc;

/// Configuration for a [`UCB1Agent`]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UCB1AgentConfig {
    /// Scale factor on the confidence interval; controls the exploration rate.
    ///
    /// A value of 0.2 is recommended by Audibert and Munos in their ICML
    /// tutorial Introduction to Bandits: Algorithms and Theory (2011).
    pub exploration_rate: f64,
}

impl UCB1AgentConfig {
    #[must_use]
    pub const fn new(exploration_rate: f64) -> Self {
        Self { exploration_rate }
    }
}

impl Default for UCB1AgentConfig {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl<OS, AS> BuildAgent<OS, AS, IntervalSpace<Reward>> for UCB1AgentConfig
where
    OS: FiniteSpace + Clone + 'static,
    AS: FiniteSpace + Clone + 'static,
{
    type Agent = UCB1Agent<OS, AS>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<
            ObservationSpace = OS,
            ActionSpace = AS,
            FeedbackSpace = IntervalSpace<Reward>,
        >,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        let observation_space = env.observation_space();
        let action_space = env.action_space();
        let IntervalSpace {
            low: Reward(r_min),
            high: Reward(r_max),
        } = env.feedback_space();
        Ok(FiniteSpaceAgent {
            agent: Arc::new(BaseUCB1Agent::new(
                observation_space.size(),
                action_space.size(),
                (r_min, r_max),
                self.exploration_rate,
            )?),
            observation_space,
            action_space,
        })
    }
}

/// A UCB1 Agent
///
/// Applies UCB1 (Auer 2002) independently to each state.
pub type UCB1Agent<OS, AS> = FiniteSpaceAgent<Arc<BaseUCB1Agent>, OS, AS>;

/// Base UCB1 Agent
///
/// Applies UCB1 (Auer 2002) independently to each state.
/// Defined for index observation and action spaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseUCB1Agent {
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
    /// Update based on an on-policy or off-policy step.
    fn step_update(&mut self, step: PartialStep<usize, usize>) {
        let scaled_reward = (step.feedback.unwrap() + self.reward_shift) * self.reward_scale_factor;

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

impl Agent<usize, usize> for Arc<BaseUCB1Agent> {
    type Actor = BaseUCB1Actor;

    fn actor(&self, mode: ActorMode) -> Self::Actor {
        BaseUCB1Actor {
            agent: self.clone(), // Arc clone
            mode,
        }
    }
}

impl BatchUpdate<usize, usize> for Arc<BaseUCB1Agent> {
    type Feedback = Reward;
    type HistoryBuffer = VecBuffer<usize, usize>;

    fn buffer(&self) -> Self::HistoryBuffer {
        VecBuffer::new()
    }

    fn min_update_size(&self) -> HistoryDataBound {
        HistoryDataBound {
            min_steps: 1,
            slack_steps: 0,
        }
    }

    fn batch_update<'a, I>(&mut self, buffers: I, _logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        let agent = Self::get_mut(self).expect("cannot update agent while actors exist");
        for buffer in buffers {
            for step in buffer.drain_steps() {
                agent.step_update(step)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseUCB1Actor {
    agent: Arc<BaseUCB1Agent>,
    mode: ActorMode,
}

impl Actor<usize, usize> for BaseUCB1Actor {
    type EpisodeState = ();
    fn initial_state(&self, _: &mut Prng) -> Self::EpisodeState {}

    fn act(&self, _: &mut Self::EpisodeState, obs: &usize, _: &mut Prng) -> usize {
        match self.mode {
            ActorMode::Training => {
                // Maximize upper confidence bound
                let log_squared_visit_count =
                    2.0 * (self.agent.state_visit_count[*obs] as f64).ln();
                let ucb = self
                    .agent
                    .state_action_count
                    .index_axis(Axis(0), *obs)
                    .mapv(|action_count| {
                        (log_squared_visit_count / (action_count as f64)).sqrt()
                            * self.agent.exploration_rate
                    })
                    + self
                        .agent
                        .state_action_mean_reward
                        .index_axis(Axis(0), *obs);
                ucb.into_iter()
                    .argmax_by(|a, b| a.partial_cmp(b).unwrap())
                    .expect("empty action space")
            }
            ActorMode::Evaluation => {
                // Maximize action count
                self.agent
                    .state_action_count
                    .index_axis(Axis(0), *obs)
                    .into_iter()
                    .argmax_by(|a, b| a.partial_cmp(b).unwrap())
                    .expect("empty action space")
            }
        }
    }
}

#[cfg(test)]
mod ucb1_agent {
    use super::super::super::testing;
    use super::*;

    #[test]
    fn learns_determinstic_bandit() {
        testing::train_deterministic_bandit(&UCB1AgentConfig::default(), 1000, 0.9);
    }
}
