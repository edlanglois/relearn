//! Generic Markov Decision Processes
use super::{CloneBuild, EnvDistribution, EnvStructure, Environment, Successor};
use crate::feedback::Reward;
use crate::logging::StatsLogger;
use crate::spaces::{IndexSpace, IntervalSpace};
use crate::Prng;
use ndarray::{Array2, Axis};
use rand::distributions::Distribution;
use rand_distr::{Dirichlet, Normal, WeightedAliasIndex};
use serde::{Deserialize, Serialize};

/// An MDP with transition and reward functions stored in lookup tables.
///
/// The environment always starts in the state with index 0.
/// There are no terminal states; episodes last forever.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabularMdp<T = WeightedAliasIndex<f32>, D = Normal<f64>> {
    /// Environment transitions table.
    ///
    /// Given state index `s` and action index `a`,
    /// `transitions[s, a] = (successor_distribution, reward_distribution)` where
    /// `successor_distribution` is the distribution of the successor state index `s'` and
    /// `reward_distribution` is the reward distribution for the step.
    pub transitions: Array2<(T, D)>,

    /// Environment discount factor.
    pub discount_factor: f64,
}

impl<T: Clone, D: Clone> CloneBuild for TabularMdp<T, D> {}

impl<T, D> EnvStructure for TabularMdp<T, D> {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(self.transitions.len_of(Axis(0)))
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.transitions.len_of(Axis(1)))
    }

    fn feedback_space(&self) -> Self::FeedbackSpace {
        // TODO: Could get a tighter bound by requiring D: Bounded
        IntervalSpace::default()
    }

    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }
}

impl<T, D> Environment for TabularMdp<T, D>
where
    T: Distribution<usize>,
    D: Distribution<f64>,
{
    type State = usize;
    type Observation = usize;
    type Action = usize;
    type Feedback = Reward;

    fn initial_state(&self, _: &mut Prng) -> Self::State {
        0
    }

    fn observe(&self, state: &Self::State, _: &mut Prng) -> Self::Observation {
        *state
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        _: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        let (successor_distribution, reward_distribution) = &self.transitions[(state, *action)];
        let next_state = successor_distribution.sample(rng);
        let reward = reward_distribution.sample(rng);
        (Successor::Continue(next_state), reward.into())
    }
}

/// Random distribution over MDPs with Dirichlet sampled transition probabilities.
///
/// * Each state-action pair has a categorical successor state distribution sampled from
///     a Dirichlet prior.
/// * Step rewards are sampled from a normal distribution with variance 1 and mean sampled
///     from a normal prior.
///
/// # Reference
/// This environment appears in the paper
/// "[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning][rl2]" by Duan et al.
///
/// [rl2]: https://arxiv.org/abs/1611.02779
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirichletRandomMdps {
    pub num_states: usize,
    pub num_actions: usize,
    pub transition_prior_dirichlet_alpha: f64,
    pub reward_prior_mean: f64,
    pub reward_prior_stddev: f64,
    pub discount_factor: f64,
}

impl CloneBuild for DirichletRandomMdps {}

impl Default for DirichletRandomMdps {
    fn default() -> Self {
        Self {
            num_states: 10,
            num_actions: 5,
            transition_prior_dirichlet_alpha: 1.0,
            reward_prior_mean: 1.0,
            reward_prior_stddev: 1.0,
            discount_factor: 1.0, // Assumes that a step limit will be placed on the episodes
        }
    }
}

impl EnvStructure for DirichletRandomMdps {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(self.num_states)
    }
    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.num_actions)
    }
    fn feedback_space(&self) -> Self::FeedbackSpace {
        IntervalSpace::new(Reward(f64::NEG_INFINITY), Reward(f64::INFINITY))
    }
    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }
}

impl EnvDistribution for DirichletRandomMdps {
    type State = <Self::Environment as Environment>::State;
    type Observation = <Self::Environment as Environment>::Observation;
    type Action = <Self::Environment as Environment>::Action;
    type Feedback = <Self::Environment as Environment>::Feedback;
    type Environment = TabularMdp;

    #[allow(clippy::cast_possible_truncation)]
    fn sample_environment(&self, rng: &mut Prng) -> Self::Environment {
        // Sample f32 values to save space since the precision of f64 shouldn't be necessary
        let dynamics_prior = Dirichlet::new_with_size(
            self.transition_prior_dirichlet_alpha as f32,
            self.num_states,
        )
        .expect("Invalid Dirichlet distribution");
        let reward_prior = Normal::new(self.reward_prior_mean, self.reward_prior_stddev)
            .expect("Invalid Normal distribution");
        let transitions = Array2::from_shape_simple_fn([self.num_states, self.num_actions], || {
            (
                WeightedAliasIndex::new(dynamics_prior.sample(rng)).unwrap(),
                Normal::new(reward_prior.sample(rng), 1.0).unwrap(),
            )
        });
        TabularMdp {
            transitions,
            discount_factor: self.discount_factor,
        }
    }
}

#[cfg(test)]
mod dirichlet_random_mdps {
    use super::super::testing;
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn run_sample() {
        let env_dist = DirichletRandomMdps::default();
        let mut rng = Prng::seed_from_u64(168);
        let env = env_dist.sample_environment(&mut rng);
        testing::check_structured_env(&env, 1000, 170);
    }

    #[test]
    fn subset_env_structure() {
        let env_dist = DirichletRandomMdps::default();
        testing::check_env_distribution_structure(&env_dist, 5);
    }
}
