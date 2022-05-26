//! Tabular agents
use super::{
    buffers::VecBuffer, finite::FiniteSpaceAgent, Actor, ActorMode, Agent, BatchUpdate, BuildAgent,
    BuildAgentError, HistoryDataBound,
};
use crate::envs::EnvStructure;
use crate::logging::StatsLogger;
use crate::simulation::{StepsIter, TransientStep};
use crate::spaces::FiniteSpace;
use crate::Prng;
use ndarray::{Array, Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Configuration of an Epsilon-Greedy Tabular Q Learning Agent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TabularQLearningAgentConfig {
    /// Probability of taking a random action.
    pub exploration_rate: f64,

    /// Initial action count for each state-action pair.
    pub initial_action_count: u64,

    /// Initial action value for each state-action pair.
    pub initial_action_value: f64,
}

impl TabularQLearningAgentConfig {
    #[must_use]
    pub const fn new(exploration_rate: f64) -> Self {
        Self {
            exploration_rate,
            initial_action_count: 0,
            initial_action_value: 0.0,
        }
    }
}

impl Default for TabularQLearningAgentConfig {
    fn default() -> Self {
        Self {
            exploration_rate: 0.2,
            initial_action_count: 0,
            initial_action_value: 0.0,
        }
    }
}

impl<OS, AS> BuildAgent<OS, AS> for TabularQLearningAgentConfig
where
    OS: FiniteSpace + Clone + 'static,
    AS: FiniteSpace + Clone + 'static,
{
    type Agent = TabularQLearningAgent<OS, AS>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        let observation_space = env.observation_space();
        let action_space = env.action_space();
        Ok(FiniteSpaceAgent {
            agent: BaseTabularQLearningAgent::new(
                observation_space.size(),
                action_space.size(),
                env.discount_factor(),
                self.exploration_rate,
            ),
            observation_space,
            action_space,
        })
    }
}

/// An epsilon-greedy tabular Q learning agent.
pub type TabularQLearningAgent<OS, AS> = FiniteSpaceAgent<BaseTabularQLearningAgent, OS, AS>;

/// Base epsilon-greedy tabular Q learning agent.
///
/// Implemented only for index observation and action spaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseTabularQLearningAgent {
    pub discount_factor: f64,
    pub exploration_rate: f64,

    state_action_counts: Array2<u64>,
    state_action_values: Arc<Array2<f64>>,
}

impl BaseTabularQLearningAgent {
    pub fn new(
        num_observations: usize,
        num_actions: usize,
        discount_factor: f64,
        exploration_rate: f64,
    ) -> Self {
        Self::from_priors(
            num_observations,
            num_actions,
            discount_factor,
            exploration_rate,
            0,
            0.0,
        )
    }

    pub fn from_priors(
        num_observations: usize,
        num_actions: usize,
        discount_factor: f64,
        exploration_rate: f64,
        prior_count: u64,
        prior_value: f64,
    ) -> Self {
        let state_action_counts = Array::from_elem((num_observations, num_actions), prior_count);
        let state_action_values = Arc::new(Array::from_elem(
            (num_observations, num_actions),
            prior_value,
        ));
        Self {
            discount_factor,
            exploration_rate,
            state_action_counts,
            state_action_values,
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

impl Agent<usize, usize> for BaseTabularQLearningAgent {
    type Actor = BaseTabularQLearningActor;

    fn actor(&self, mode: ActorMode) -> Self::Actor {
        BaseTabularQLearningActor {
            state_action_values: Arc::clone(&self.state_action_values),
            exploration_rate: self.exploration_rate,
            mode,
        }
    }
}

impl BaseTabularQLearningAgent {
    /// Update based on an on-policy or off-policy step.
    fn step_update(&mut self, step: TransientStep<usize, usize>) {
        let discounted_next_value = match step.next.as_ref().into_inner() {
            None => 0.0,
            Some(&next_observation) => {
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
        let weight = (self.state_action_counts[idx] as f64).recip();
        let state_action_values = Arc::get_mut(&mut self.state_action_values)
            .expect("cannot update agent while actors exist");
        state_action_values[idx] *= 1.0 - weight;
        state_action_values[idx] += weight * value;
    }
}

impl BatchUpdate<usize, usize> for BaseTabularQLearningAgent {
    type HistoryBuffer = VecBuffer<usize, usize>;

    fn buffer(&self) -> Self::HistoryBuffer {
        VecBuffer::new()
    }

    fn min_update_size(&self) -> HistoryDataBound {
        HistoryDataBound {
            min_steps: 1,
            slack_steps: 0, // does not matter if the episode has ended
        }
    }

    fn batch_update<'a, I>(&mut self, buffers: I, _logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        for buffer in buffers {
            buffer
                .drain_steps()
                .for_each_transient(|step| self.step_update(step));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseTabularQLearningActor {
    state_action_values: Arc<Array2<f64>>,
    exploration_rate: f64,
    mode: ActorMode,
}

impl Actor<usize, usize> for BaseTabularQLearningActor {
    type EpisodeState = ();

    fn initial_state(&self, _: &mut Prng) -> Self::EpisodeState {}

    fn act(&self, _: &mut Self::EpisodeState, observation: &usize, rng: &mut Prng) -> usize {
        if self.mode == ActorMode::Training && rng.gen::<f64>() < self.exploration_rate {
            let (_, num_actions) = self.state_action_values.dim();
            rng.gen_range(0..num_actions)
        } else {
            self.state_action_values
                .index_axis(Axis(0), *observation)
                .argmax()
                .expect("action space must be non-empty")
        }
    }
}

#[cfg(test)]
mod tabular_q_learning {
    use super::super::{testing, BuildAgent};
    use super::*;
    use crate::envs::{DeterministicBandit, Environment};
    use crate::simulation::{self, SimSeed};
    use rand::SeedableRng;

    #[test]
    fn learns_determinstic_bandit() {
        testing::train_deterministic_bandit(&TabularQLearningAgentConfig::default(), 1000, 0.9);
    }

    #[test]
    fn explore_exploit() {
        let mut env_rng = Prng::seed_from_u64(210);
        let mut agent_rng = Prng::seed_from_u64(211);

        let env = DeterministicBandit::from_values(vec![0.0, 1.0]);
        let config = TabularQLearningAgentConfig::new(0.95);
        let mut agent = config.build_agent(&env, &mut agent_rng).unwrap();

        simulation::train_serial(&mut agent, &env, 100, &mut env_rng, &mut agent_rng, &mut ());

        // The training mode explores
        let mut train_action_1_count = 0;
        for step in (&env)
            .run(agent.actor(ActorMode::Training), SimSeed::Root(216), ())
            .take(1000)
        {
            if step.action == 1 {
                train_action_1_count += 1;
            }
        }
        assert!(train_action_1_count > 300);
        assert!(train_action_1_count < 700);

        // Evaluation mode exploits
        let mut eval_action_1_count = 0;
        for step in (&env)
            .run(agent.actor(ActorMode::Evaluation), SimSeed::Root(224), ())
            .take(1000)
        {
            if step.action == 1 {
                eval_action_1_count += 1;
            }
        }
        assert!(eval_action_1_count > 900);
    }
}
