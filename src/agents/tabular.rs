//! Tabular agents
use ndarray::{Array, Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::Rng;

use super::{Actor, Agent, Step};
use crate::spaces::FiniteSpace;

pub struct TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    pub observation_space: OS,
    pub action_space: AS,
    pub discount_factor: f32,
    pub exploration_rate: f32,
    pub state_action_counts: Array2<u32>,
    pub state_action_values: Array2<f32>,
}

impl<OS, AS> TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    pub fn new(
        observation_space: OS,
        action_space: AS,
        discount_factor: f32,
        exploration_rate: f32,
    ) -> Self {
        let num_observations = observation_space.len();
        let num_actions = action_space.len();
        let state_action_counts = Array::from_elem((num_observations, num_actions), 0);
        let state_action_values = Array::from_elem((num_observations, num_actions), 0.0);
        Self {
            observation_space,
            action_space,
            discount_factor,
            exploration_rate,
            state_action_counts,
            state_action_values,
        }
    }
}

impl<OS, AS, R> Actor<OS::Element, AS::Element, R> for TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
    R: Rng,
{
    fn act(&mut self, observation: &OS::Element, _new_episode: bool, rng: &mut R) -> AS::Element {
        if rng.gen::<f32>() < self.exploration_rate {
            self.action_space.sample(rng)
        } else {
            let obs_idx = self.observation_space.index_of(observation);
            let act_idx = self
                .state_action_values
                .index_axis(Axis(0), obs_idx)
                .argmax()
                .unwrap();
            self.action_space.index(act_idx)
        }
    }
}

impl<OS, AS, R> Agent<OS::Element, AS::Element, R> for TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
    R: Rng,
{
    fn update(&mut self, step: Step<OS::Element, AS::Element>) {
        let obs_idx = self.observation_space.index_of(&step.observation);
        let act_idx = self.action_space.index_of(&step.action);

        let discounted_next_value = match step.next_observation {
            None => 0.0,
            Some(next_observation) => {
                let next_obs_idx = self.observation_space.index_of(&next_observation);
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
        let weight = (self.state_action_counts[idx] as f32).recip();
        self.state_action_values[idx] *= 1.0 - weight;
        self.state_action_values[idx] += weight * value;
    }
}
