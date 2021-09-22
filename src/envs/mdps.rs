//! Generic Markov Decision Processes
use super::{CloneBuild, EnvStructure, Mdp, PomdpDistribution};
use crate::spaces::IndexSpace;
use ndarray::{Array2, Axis};
use rand::{distributions::Distribution, rngs::StdRng};
use rand_distr::{Dirichlet, Normal, WeightedAliasIndex};

/// An MDP with transition and reward functions stored in lookup tables.
///
/// The environment always starts in the state with index 0.
/// There are no terminal states; episodes last forever.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularMdp<T = WeightedAliasIndex<f32>, R = Normal<f64>> {
    /// Environment transitions table.
    ///
    /// Given state index `s` and action index `a`,
    /// `transitions[s, a] = (successor_distribution, reward_distribution)` where
    /// `successor_distribution` is the distribution of the successor state index `s'` and
    /// `reward_distribution` is the reward distribution for the step.
    pub transitions: Array2<(T, R)>,

    /// Environment discount factor.
    pub discount_factor: f64,
}

impl<T: Clone, R: Clone> CloneBuild for TabularMdp<T, R> {}

impl<T, R> EnvStructure for TabularMdp<T, R> {
    type ObservationSpace = IndexSpace;
    type ActionSpace = IndexSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(self.transitions.len_of(Axis(0)))
    }

    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.transitions.len_of(Axis(1)))
    }

    fn reward_range(&self) -> (f64, f64) {
        // TODO: Could get a tighter bound by requiring R: Bounded
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }
}

impl<T, R> Mdp for TabularMdp<T, R>
where
    T: Distribution<usize>,
    R: Distribution<f64>,
{
    type State = usize;
    type Action = usize;

    fn initial_state(&self, _rng: &mut StdRng) -> Self::State {
        0
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut StdRng,
    ) -> (Option<Self::State>, f64, bool) {
        let (successor_distribution, reward_distribution) = &self.transitions[(state, *action)];
        let next_state = successor_distribution.sample(rng);
        let reward = reward_distribution.sample(rng);
        (Some(next_state), reward, false)
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
/// [rl2]: https://arxiv.org/pdf/1611.02779
#[derive(Debug, Clone, Copy, PartialEq)]
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

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(self.num_states)
    }
    fn action_space(&self) -> Self::ActionSpace {
        IndexSpace::new(self.num_actions)
    }
    fn reward_range(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }
}

impl PomdpDistribution for DirichletRandomMdps {
    type Pomdp = TabularMdp;

    #[allow(clippy::cast_possible_truncation)]
    fn sample_pomdp(&self, rng: &mut StdRng) -> Self::Pomdp {
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
        let mut rng = StdRng::seed_from_u64(168);
        let env = env_dist.sample_pomdp(&mut rng);
        testing::run_pomdp(env, 1000, 170);
    }

    #[test]
    fn subset_env_structure() {
        let env_dist = DirichletRandomMdps::default();
        testing::check_env_distribution_structure(&env_dist, 5);
    }
}
