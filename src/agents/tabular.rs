//! Tabular agents
use super::{Actor, ActorMode, Agent, AgentBuilder, BuildAgentError, SetActorMode, Step};
use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::spaces::{FiniteSpace, SampleSpace};
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

impl<E> AgentBuilder<TabularQLearningAgent<E::ObservationSpace, E::ActionSpace>, E>
    for TabularQLearningAgentConfig
where
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: FiniteSpace,
    <E as EnvStructure>::ActionSpace: FiniteSpace,
{
    fn build_agent(
        &self,
        env: &E,
        seed: u64,
    ) -> Result<TabularQLearningAgent<E::ObservationSpace, E::ActionSpace>, BuildAgentError> {
        Ok(TabularQLearningAgent::new(
            env.observation_space(),
            env.action_space(),
            env.discount_factor(),
            self.exploration_rate,
            seed,
        ))
    }
}

/// An epsilon-greedy tabular Q learning.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularQLearningAgent<OS, AS> {
    pub observation_space: OS,
    pub action_space: AS,
    pub discount_factor: f64,
    pub exploration_rate: f64,
    pub state_action_counts: Array2<u32>,
    pub state_action_values: Array2<f64>,
    pub mode: ActorMode,

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
            mode: ActorMode::Training,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl<OS, AS> fmt::Display for TabularQLearningAgent<OS, AS>
where
    OS: fmt::Display,
    AS: fmt::Display,
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
        if self.mode == ActorMode::Training && self.rng.gen::<f64>() < self.exploration_rate {
            // Random exploration with probability `exploration_rate` when in training mode
            self.action_space.sample(&mut self.rng)
        } else {
            let obs_idx = self.observation_space.to_index(observation);
            let act_idx = self
                .state_action_values
                .index_axis(Axis(0), obs_idx)
                .argmax()
                .unwrap();
            self.action_space.from_index(act_idx).unwrap()
        }
    }
}

impl<OS, AS> TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace,
{
    fn indexed_update(&mut self, step: &Step<usize, usize>, _logger: &mut dyn TimeSeriesLogger) {
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

impl<OS, AS> Agent<OS::Element, AS::Element> for TabularQLearningAgent<OS, AS>
where
    OS: FiniteSpace,
    AS: FiniteSpace + SampleSpace,
{
    fn update(&mut self, step: Step<OS::Element, AS::Element>, logger: &mut dyn TimeSeriesLogger) {
        self.indexed_update(
            &super::indexed_step(&step, &self.observation_space, &self.action_space),
            logger,
        )
    }
}

impl<OS, AS> SetActorMode for TabularQLearningAgent<OS, AS> {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.mode = mode;
    }
}

#[cfg(test)]
mod tabular_q_learning {
    use super::super::testing;
    use super::*;
    use crate::envs::{DeterministicBandit, IntoStateful};
    use crate::simulation;
    use crate::simulation::hooks::{IndexedActionCounter, StepLimit};

    #[test]
    fn learns_determinstic_bandit() {
        let config = TabularQLearningAgentConfig::default();
        testing::train_deterministic_bandit(
            |env_structure| config.build_agent(env_structure, 0).unwrap(),
            1000,
            0.9,
        );
    }

    #[test]
    fn explore_exploit() {
        let mut env = DeterministicBandit::from_values(vec![0.0, 1.0]).into_stateful(0);

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
