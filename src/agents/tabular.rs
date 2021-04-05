//! Tabular agents
use super::{Actor, Agent, AgentBuilder, NewAgentError, Step};
use crate::envs::EnvStructure;
use crate::logging::Logger;
use crate::spaces::{FiniteSpace, SampleSpace};
use ndarray::{Array, Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

/// Configuration of an Epsilon-Greedy Tabular Q Learning Agent.
#[derive(Debug)]
pub struct TabularQLearningAgentConfig {
    /// Probability of taking a random action.
    pub exploration_rate: f64,
}

impl TabularQLearningAgentConfig {
    pub fn new(exploration_rate: f64) -> Self {
        Self { exploration_rate }
    }
}

impl Default for TabularQLearningAgentConfig {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl<OS: FiniteSpace, AS: FiniteSpace + SampleSpace> AgentBuilder<OS, AS>
    for TabularQLearningAgentConfig
{
    type Agent = TabularQLearningAgent<OS, AS>;

    fn build(&self, es: EnvStructure<OS, AS>, seed: u64) -> Result<Self::Agent, NewAgentError> {
        Ok(Self::Agent::new(
            es.observation_space,
            es.action_space,
            es.discount_factor,
            self.exploration_rate,
            seed,
        ))
    }
}

/// An epsilon-greedy tabular Q learning.
#[derive(Debug)]
pub struct TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    pub observation_space: OS,
    pub action_space: AS,
    pub discount_factor: f64,
    pub exploration_rate: f64,
    pub state_action_counts: Array2<u32>,
    pub state_action_values: Array2<f64>,

    rng: StdRng,
}

impl<OS, AS> TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    pub fn new(
        observation_space: OS,
        action_space: AS,
        discount_factor: f64,
        exploration_rate: f64,
        seed: u64,
    ) -> Self {
        let num_observations = observation_space.size();
        let num_actions = action_space.size();
        let state_action_counts = Array::from_elem((num_observations, num_actions), 0);
        let state_action_values = Array::from_elem((num_observations, num_actions), 0.0);
        Self {
            observation_space,
            action_space,
            discount_factor,
            exploration_rate,
            state_action_counts,
            state_action_values,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<OS, AS> fmt::Display for TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace + fmt::Display,
    AS: FiniteSpace + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TabularQLearningAgent({}, {}, {}, {})",
            self.observation_space, self.action_space, self.discount_factor, self.exploration_rate
        )
    }
}

impl<OS, AS> Actor<OS::Element, AS::Element> for TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace + SampleSpace,
{
    fn act(&mut self, observation: &OS::Element, _new_episode: bool) -> AS::Element {
        let obs_idx = self.observation_space.to_index(observation);
        let act_idx = self
            .state_action_values
            .index_axis(Axis(0), obs_idx)
            .argmax()
            .unwrap();
        self.action_space.from_index(act_idx).unwrap()
    }
}

impl<OS, AS> Agent<OS::Element, AS::Element> for TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace + SampleSpace,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        if self.rng.gen::<f64>() < self.exploration_rate {
            self.action_space.sample(&mut self.rng)
        } else {
            Actor::act(self, observation, new_episode)
        }
    }

    fn update(&mut self, step: Step<OS::Element, AS::Element>, _logger: &mut dyn Logger) {
        let obs_idx = self.observation_space.to_index(&step.observation);
        let act_idx = self.action_space.to_index(&step.action);

        let discounted_next_value = match step.next_observation {
            None => 0.0,
            Some(next_observation) => {
                let next_obs_idx = self.observation_space.to_index(&next_observation);
                self.state_action_values
                    .index_axis(Axis(0), next_obs_idx)
                    .max()
                    .unwrap()
                    * self.discount_factor
            }
        };
        let idx = (obs_idx, act_idx);
        self.state_action_counts[idx] += 1;

        let value = step.reward + discounted_next_value;
        let weight = (self.state_action_counts[idx] as f64).recip();
        self.state_action_values[idx] *= 1.0 - weight;
        self.state_action_values[idx] += weight * value;
    }
}

#[cfg(test)]
mod tabular_q_learning {
    use super::super::testing;
    use super::*;
    use crate::envs::{AsStateful, DeterministicBandit, StatefulEnvironment};
    use crate::simulation;
    use crate::simulation::hooks::{IndexedActionCounter, StepLimit};

    #[test]
    fn learns_determinstic_bandit() {
        let config = TabularQLearningAgentConfig::default();
        testing::train_deterministic_bandit(
            |env_structure| config.build(env_structure, 0).unwrap(),
            1000,
            0.9,
        );
    }

    #[test]
    fn explore_exploit() {
        let mut env = DeterministicBandit::from_values(vec![0.0, 1.0]).as_stateful(0);
        let env_structure = env.structure();
        let action_space = env_structure.action_space.clone();

        // The agent explores
        let config = TabularQLearningAgentConfig::new(0.95);
        let mut agent = config.build(env_structure, 0).unwrap();
        let mut explore_hooks = (
            IndexedActionCounter::new(action_space.clone()),
            StepLimit::new(1000),
        );
        simulation::run_agent(&mut env, &mut agent, &mut (), &mut explore_hooks);
        let action_1_count = explore_hooks.0.counts[1];
        assert!(action_1_count > 300);
        assert!(action_1_count < 700);

        // The actor exploits
        let mut exploit_hooks = (
            IndexedActionCounter::new(action_space),
            StepLimit::new(1000),
        );
        simulation::run_actor(&mut env, &mut agent, &mut (), &mut exploit_hooks);
        assert!(exploit_hooks.0.counts[1] > 900);
    }
}
