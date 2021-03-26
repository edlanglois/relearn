//! Tabular agents
use super::{Actor, Agent, Step};
use crate::spaces::FiniteSpace;
use crate::utils::iter::ArgMaxBy;
use ndarray::{Array, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

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
    AS: FiniteSpace,
{
    fn act(&mut self, observation: &OS::Element, _new_episode: bool) -> AS::Element {
        if self.rng.gen::<f64>() < self.exploration_rate {
            self.action_space.sample(&mut self.rng)
        } else {
            let obs_idx = self.observation_space.to_index(observation);
            let act_idx = self
                .state_action_values
                .index_axis(Axis(0), obs_idx)
                .iter()
                .argmax_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            self.action_space.from_index(act_idx).unwrap()
        }
    }
}

impl<OS, AS> Agent<OS::Element, AS::Element> for TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn update(&mut self, step: Step<OS::Element, AS::Element>) {
        let obs_idx = self.observation_space.to_index(&step.observation);
        let act_idx = self.action_space.to_index(&step.action);

        let discounted_next_value = match step.next_observation {
            None => 0.0,
            Some(next_observation) => {
                let next_obs_idx = self.observation_space.to_index(&next_observation);
                self.state_action_values
                    .index_axis(Axis(0), next_obs_idx)
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
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
mod tests {
    use super::super::testing;
    use super::*;

    #[test]
    fn train_tabular_q_learning_train() {
        testing::train_deterministic_bandit(
            |env_structure| {
                TabularQLearningAgent::new(
                    env_structure.observation_space,
                    env_structure.action_space,
                    env_structure.discount_factor,
                    0.1,
                    0,
                )
            },
            1000,
            0.9,
        );
    }

    #[test]
    #[should_panic]
    fn train_tabular_q_learning_exploration() {
        testing::train_deterministic_bandit(
            |env_structure| {
                TabularQLearningAgent::new(
                    env_structure.observation_space,
                    env_structure.action_space,
                    env_structure.discount_factor,
                    0.95,
                    0,
                )
            },
            1000,
            0.9,
        );
    }
}
