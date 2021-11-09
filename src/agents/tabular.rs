//! Tabular agents
use super::{
    Actor, ActorMode, Agent, BuildAgentError, BuildIndexAgent, FiniteSpaceAgent, OffPolicyAgent,
    SetActorMode, Step, SyncParams, SyncParamsError,
};
use crate::logging::TimeSeriesLogger;
use ndarray::{Array, Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

/// Configuration of an Epsilon-Greedy Tabular Q Learning Agent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TabularQLearningAgentConfig {
    /// Probability of taking a random action.
    pub exploration_rate: f64,
}

impl TabularQLearningAgentConfig {
    pub const fn new(exploration_rate: f64) -> Self {
        Self { exploration_rate }
    }
}

impl Default for TabularQLearningAgentConfig {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl BuildIndexAgent for TabularQLearningAgentConfig {
    type Agent = BaseTabularQLearningAgent;

    fn build_index_agent(
        &self,
        num_observations: usize,
        num_actions: usize,
        _reward_range: (f64, f64),
        discount_factor: f64,
        seed: u64,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(BaseTabularQLearningAgent::new(
            num_observations,
            num_actions,
            discount_factor,
            self.exploration_rate,
            seed,
        ))
    }
}

/// An epsilon-greedy tabular Q learning agent.
pub type TabularQLearningAgent<OS, AS> = FiniteSpaceAgent<BaseTabularQLearningAgent, OS, AS>;

/// Base epsilon-greedy tabular Q learning agent.
///
/// Implemented only for index observation and action spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct BaseTabularQLearningAgent {
    pub discount_factor: f64,
    pub exploration_rate: f64,
    pub mode: ActorMode,

    state_action_counts: Array2<u32>,
    state_action_values: Array2<f64>,
    rng: StdRng,
}

impl BaseTabularQLearningAgent {
    pub fn new(
        num_observations: usize,
        num_actions: usize,
        discount_factor: f64,
        exploration_rate: f64,
        seed: u64,
    ) -> Self {
        let state_action_counts = Array::from_elem((num_observations, num_actions), 0);
        let state_action_values = Array::from_elem((num_observations, num_actions), 0.0);
        Self {
            discount_factor,
            exploration_rate,
            state_action_counts,
            state_action_values,
            mode: ActorMode::Training,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl fmt::Display for BaseTabularQLearningAgent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BaseTabularQLearningAgent(γ={}, ϵ={})",
            self.discount_factor, self.exploration_rate
        )
    }
}

impl Actor<usize, usize> for BaseTabularQLearningAgent {
    fn act(&mut self, observation: &usize, _new_episode: bool) -> usize {
        if self.mode == ActorMode::Training && self.rng.gen::<f64>() < self.exploration_rate {
            // Random exploration with probability `exploration_rate` when in training mode
            let (_, num_actions) = self.state_action_counts.dim();
            self.rng.gen_range(0..num_actions)
        } else {
            self.state_action_values
                .index_axis(Axis(0), *observation)
                .argmax()
                .unwrap()
        }
    }
}

impl Agent<usize, usize> for BaseTabularQLearningAgent {
    fn update(&mut self, step: Step<usize, usize>, _logger: &mut dyn TimeSeriesLogger) {
        let discounted_next_value = match step.next_observation {
            None => 0.0,
            Some(next_observation) => {
                self.state_action_values
                    .index_axis(Axis(0), next_observation)
                    .max()
                    .unwrap()
                    * self.discount_factor
            }
        };
        let idx = (step.observation, step.action);
        self.state_action_counts[idx] += 1;

        let value = step.reward + discounted_next_value;
        let weight = f64::from(self.state_action_counts[idx]).recip();
        self.state_action_values[idx] *= 1.0 - weight;
        self.state_action_values[idx] += weight * value;
    }
}

impl OffPolicyAgent for BaseTabularQLearningAgent {}

impl SetActorMode for BaseTabularQLearningAgent {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.mode = mode;
    }
}

impl SyncParams for BaseTabularQLearningAgent {
    fn sync_params(&mut self, target: &Self) -> Result<(), SyncParamsError> {
        if self.state_action_counts.raw_dim() == target.state_action_counts.raw_dim() {
            self.state_action_counts.assign(&target.state_action_counts);
            self.state_action_values.assign(&target.state_action_values);
            Ok(())
        } else {
            Err(SyncParamsError::IncompatibleParams)
        }
    }
}

#[cfg(test)]
mod tabular_q_learning {
    use super::super::{testing, BuildAgent};
    use super::*;
    use crate::envs::{DeterministicBandit, EnvStructure, IntoEnv};
    use crate::simulation;
    use crate::simulation::hooks::{IndexedActionCounter, StepLimit};

    #[test]
    fn learns_determinstic_bandit() {
        let config = TabularQLearningAgentConfig::default();
        testing::train_deterministic_bandit(|env| config.build_agent(env, 0).unwrap(), 1000, 0.9);
    }

    #[test]
    fn explore_exploit() {
        let mut env = DeterministicBandit::from_values(vec![0.0, 1.0]).into_env(0);

        // The initial mode explores
        let config = TabularQLearningAgentConfig::new(0.95);
        let mut agent = config.build_agent(&env, 0).unwrap();
        let mut explore_hooks = (
            IndexedActionCounter::new(env.action_space()),
            StepLimit::new(1000),
        );
        simulation::run_agent(&mut env, &mut agent, &mut explore_hooks, &mut ());
        let action_1_count = explore_hooks.0.counts[1];
        assert!(action_1_count > 300);
        assert!(action_1_count < 700);

        // Release mode exploits
        agent.set_actor_mode(ActorMode::Release);
        let mut exploit_hooks = (
            IndexedActionCounter::new(env.action_space()),
            StepLimit::new(1000),
        );
        simulation::run_actor(&mut env, &mut agent, &mut exploit_hooks, &mut ());
        assert!(exploit_hooks.0.counts[1] > 900);
    }
}
