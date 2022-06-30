//! Meta reinforcement learning environment.
use super::{
    BuildEnv, BuildEnvDist, BuildEnvError, EnvDistribution, EnvStructure, Environment, Successor,
};
use crate::logging::StatsLogger;
use crate::spaces::{BooleanSpace, OptionSpace, ProductSpace, Space};
use crate::Prng;
use serde::{Deserialize, Serialize};

/// A meta reinforcement learning environment that treats RL itself as an environment.
///
/// An episode in this meta environment is called a "Trial" and consists of
/// several episodes from the inner environment.
/// A new inner environment with a different structure seed is sampled for each Trial.
/// A meta episode ends when a fixed number of inner episodes have been completed.
///
/// The step metadata from the inner environment are embedded as observations.
///
/// # Observations
/// A [`MetaObservation`]. Consists of the inner observation, the previous step action and
/// feedback, and whether the inner episode is done.
///
/// # Actions
/// The action space is the same as the action space of the inner environments.
/// Actions are forwarded to the inner environment except when the current state is the last state
/// of the inner episode (`episode_done == true`).
/// In that case, the provided action is ignored and the next state will be the start of a new
/// inner episode.
///
/// # Feedback
/// The feedback is the same as the inner feedback.
///
/// # States
/// The state ([`MetaState`]) consists of an inner environment instance,
/// an inner environment state, an episode index within the trial, and details of the most recent
/// inner step within the episode.
///
/// # Reference
/// This meta environment design is roughly consistent with the structure used in the paper
/// "[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning][rl2]" by Duan et al.
///
/// [rl2]: https://arxiv.org/abs/1611.02779
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetaEnv<E> {
    /// Environment distribution from which each trial's episode is sampled.
    pub env_distribution: E,
    /// Number of inner episodes per trial.
    pub episodes_per_trial: u64,
}

impl<E> MetaEnv<E> {
    pub const fn new(env_distribution: E, episodes_per_trial: u64) -> Self {
        Self {
            env_distribution,
            episodes_per_trial,
        }
    }
}

impl<E: Default> Default for MetaEnv<E> {
    fn default() -> Self {
        Self {
            env_distribution: E::default(),
            episodes_per_trial: 10,
        }
    }
}

impl<EC> BuildEnv for MetaEnv<EC>
where
    EC: BuildEnvDist,
    EC::Action: Copy,
    EC::Feedback: Default,
{
    type Observation = <Self::Environment as Environment>::Observation;
    type Action = <Self::Environment as Environment>::Action;
    type Feedback = <Self::Environment as Environment>::Feedback;
    type ObservationSpace = <Self::Environment as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::Environment as EnvStructure>::ActionSpace;
    type FeedbackSpace = <Self::Environment as EnvStructure>::FeedbackSpace;
    type Environment = MetaEnv<EC::EnvDistribution>;

    fn build_env(&self, _: &mut Prng) -> Result<Self::Environment, BuildEnvError> {
        Ok(MetaEnv::new(
            self.env_distribution.build_env_dist(),
            self.episodes_per_trial,
        ))
    }
}

impl<E> EnvStructure for MetaEnv<E>
where
    E: EnvStructure,
{
    type ObservationSpace =
        MetaObservationSpace<E::ObservationSpace, E::ActionSpace, E::FeedbackSpace>;
    type ActionSpace = E::ActionSpace;
    type FeedbackSpace = E::FeedbackSpace;

    fn observation_space(&self) -> Self::ObservationSpace {
        MetaObservationSpace::from_inner_env(&self.env_distribution)
    }
    fn action_space(&self) -> Self::ActionSpace {
        self.env_distribution.action_space()
    }
    fn feedback_space(&self) -> Self::FeedbackSpace {
        self.env_distribution.feedback_space()
    }
    fn discount_factor(&self) -> f64 {
        self.env_distribution.discount_factor()
    }
}

impl<E> MetaEnv<E>
where
    E: EnvStructure,
{
    /// View the structure of the inner environment.
    pub const fn inner_structure(&self) -> InnerEnvStructure<&Self> {
        InnerEnvStructure::new(self)
    }
}

impl<E> Environment for MetaEnv<E>
where
    E: EnvDistribution,
    <E::Environment as Environment>::Action: Copy,
    <E::Environment as Environment>::Feedback: Default,
{
    type State = MetaState<E::Environment>;
    type Observation = MetaObservation<
        <E::Environment as Environment>::Observation,
        <E::Environment as Environment>::Action,
        <E::Environment as Environment>::Feedback,
    >;
    type Action = <E::Environment as Environment>::Action;
    type Feedback = <E::Environment as Environment>::Feedback;

    fn initial_state(&self, rng: &mut Prng) -> Self::State {
        // Sample a new inner environment.
        let inner_env = self.env_distribution.sample_environment(rng);
        let inner_state = inner_env.initial_state(rng);
        MetaState {
            inner_env,
            inner_successor: Successor::Continue(inner_state),
            episode_index: 0,
            prev_step_obs: None,
        }
    }

    fn observe(&self, state: &Self::State, rng: &mut Prng) -> Self::Observation {
        let inner_successor_obs = state
            .inner_successor
            .as_ref()
            .map(|s| state.inner_env.observe(s, rng));
        let episode_done = inner_successor_obs.episode_done();
        MetaObservation {
            inner_observation: inner_successor_obs.into_inner(),
            prev_step: state.prev_step_obs.clone(),
            episode_done,
        }
    }

    fn step(
        &self,
        state: Self::State,
        action: &Self::Action,
        rng: &mut Prng,
        logger: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        match state.inner_successor {
            Successor::Continue(prev_inner_state) => {
                // Take a step in the inner episode
                let (inner_successor, inner_feedback) =
                    state.inner_env.step(prev_inner_state, action, rng, logger);

                let mut new_state = MetaState {
                    inner_env: state.inner_env,
                    inner_successor,
                    episode_index: state.episode_index,
                    prev_step_obs: Some(InnerStepObs {
                        action: *action,
                        feedback: inner_feedback.clone(),
                    }),
                };

                if new_state.inner_successor.episode_done() {
                    new_state.episode_index += 1;
                    if new_state.episode_index >= self.episodes_per_trial {
                        // Completed the last inner episode of the trial.
                        // This is treated as an abrupt cut-off of a theorically infinite sequence
                        // of inner episodes so the meta state is not treated as terminal.
                        return (Successor::Interrupt(new_state), inner_feedback);
                    }
                }

                (Successor::Continue(new_state), inner_feedback)
            }
            _ => {
                // The inner state ended the episode.
                // Ignore the action and start a new inner episode.
                let inner_state = state.inner_env.initial_state(rng);
                let state = MetaState {
                    inner_env: state.inner_env,
                    inner_successor: Successor::Continue(inner_state),
                    episode_index: state.episode_index,
                    prev_step_obs: None,
                };
                (Successor::Continue(state), Default::default())
            }
        }
    }
}

// # Meta Environment Types

/// Observation of a completed inner step in a meta environment.
///
/// The agent is expected to remember the inner observation if necessary, so it is not included.
/// The agent is not expected to have a mechanism to remember its own actions, since actions only
/// ought to matter to the extent that they affect the resulting state.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct InnerStepObs<A, F> {
    /// Action selected by the agent on the step.
    pub action: A,
    /// Feedback for this step.
    pub feedback: F,
}

/// Observation space for [`InnerStepObs`].
#[derive(Debug, Copy, Clone, PartialEq, ProductSpace)]
#[element(InnerStepObs<AS::Element, FS::Element>)]
pub struct InnerStepObsSpace<AS, FS> {
    pub action: AS,
    pub feedback: FS,
}

impl<AS, FS> InnerStepObsSpace<AS, FS> {
    /// Construct a step observation space from an inner environment structure
    fn from_inner_env<E>(env: &E) -> Self
    where
        E: EnvStructure<ActionSpace = AS, FeedbackSpace = FS> + ?Sized,
    {
        Self {
            action: env.action_space(),
            feedback: env.feedback_space(),
        }
    }
}

/// An observation from a meta enviornment.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MetaObservation<O, A, F> {
    /// The current inner observation on which the next step will act.
    ///
    /// Is `None` if this is a terminal state, in which case `episode_done` must be `True`.
    pub inner_observation: Option<O>,

    /// Observation of the previous inner step.
    ///
    /// Is `None` if this is the first step of an inner episode.
    pub prev_step: Option<InnerStepObs<A, F>>,

    /// Whether the previous step ended the inner episode.
    pub episode_done: bool,
}

/// [`MetaEnv`] observation space for element [`MetaObservation`].
#[derive(Debug, Copy, Clone, PartialEq, ProductSpace)]
#[element(MetaObservation<OS::Element, AS::Element, FS::Element>)]
pub struct MetaObservationSpace<OS, AS, FS> {
    pub inner_observation: OptionSpace<OS>,
    pub prev_step: OptionSpace<InnerStepObsSpace<AS, FS>>,
    pub episode_done: BooleanSpace,
}

impl<OS, AS, FS> MetaObservationSpace<OS, AS, FS> {
    /// Construct a meta observation space from an inner environment structure
    fn from_inner_env<E>(env: &E) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS, FeedbackSpace = FS> + ?Sized,
    {
        Self {
            inner_observation: OptionSpace::new(env.observation_space()),
            prev_step: OptionSpace::new(InnerStepObsSpace::from_inner_env(env)),
            episode_done: BooleanSpace,
        }
    }
}

/// The state of a [`MetaEnv`].
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MetaState<E: Environment> {
    /// An instance of the inner environment (sampled for this trial).
    inner_env: E,
    /// The upcoming inner environment state.
    inner_successor: Successor<E::State>,
    /// The inner episode index within the current trial.
    episode_index: u64,
    /// Observation of the previous step of this inner episode.
    prev_step_obs: Option<InnerStepObs<E::Action, E::Feedback>>,
}

/// Wrapper that provides the inner environment structure of a meta environment ([`MetaEnv`]).
///
/// Implements [`EnvStructure`] according to the inner environment structure.
/// Can also be used via `MetaEnv::inner_structure`.
///
/// # Example
///
///     use relearn::envs::{InnerEnvStructure, MetaEnv, OneHotBandits, StoredEnvStructure};
///
///     let base_env = OneHotBandits::default();
///     let meta_env = MetaEnv::new(base_env, 10);
///
///     let base_structure = StoredEnvStructure::from(&base_env);
///     let meta_inner_structure = StoredEnvStructure::from(&InnerEnvStructure::new(&meta_env));
///
///     assert_eq!(base_structure, meta_inner_structure);
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InnerEnvStructure<T>(T);

impl<T> InnerEnvStructure<T> {
    pub const fn new(inner_env: T) -> Self {
        Self(inner_env)
    }
}

impl<T, OS, AS, FS> EnvStructure for InnerEnvStructure<T>
where
    T: EnvStructure<
        ObservationSpace = MetaObservationSpace<OS, AS, FS>,
        ActionSpace = AS,
        FeedbackSpace = FS,
    >,
    OS: Space,
    AS: Space,
    FS: Space,
{
    type ObservationSpace = OS;
    type ActionSpace = AS;
    type FeedbackSpace = FS;

    fn observation_space(&self) -> Self::ObservationSpace {
        self.0.observation_space().inner_observation.inner
    }
    fn action_space(&self) -> Self::ActionSpace {
        self.0.action_space()
    }
    fn feedback_space(&self) -> Self::FeedbackSpace {
        self.0.feedback_space()
    }
    fn discount_factor(&self) -> f64 {
        self.0.discount_factor()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Comparing exact reward values; 0.0 or 1.0 without error
mod meta_env_bandits {
    use super::super::testing;
    use super::*;
    use crate::feedback::Reward;
    use rand::SeedableRng;

    #[test]
    fn build_meta_env() {
        let config = MetaEnv::new(testing::RoundRobinDeterministicBandits::new(2), 3);
        let _env = config.build_env(&mut Prng::seed_from_u64(0)).unwrap();
    }

    #[test]
    fn run_meta_env() {
        let env = MetaEnv::new(testing::RoundRobinDeterministicBandits::new(2), 3);
        testing::check_structured_env(&env, 1000, 0);
    }

    #[test]
    fn meta_env_expected_steps() {
        let env = MetaEnv::new(testing::RoundRobinDeterministicBandits::new(2), 3);
        let mut rng = Prng::seed_from_u64(0);

        // Trial 0; Ep 0; Init
        let state = env.initial_state(&mut rng);
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: Some(()),
                prev_step: None,
                episode_done: false
            }
        );

        // Trial 0; Ep 0; Step 0
        // Take action 0 and get 1 reward
        // Inner state is terminal.
        let (successor, feedback) = env.step(state, &0, &mut rng, &mut ());
        assert_eq!(feedback, Reward(1.0));
        let state = successor.into_continue().unwrap();
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: None,
                prev_step: Some(InnerStepObs {
                    action: 0,
                    feedback: Reward(1.0)
                }),
                episode_done: true
            }
        );

        // Trial 0; Ep 1; Init.
        // The action is ignored and a new inner episode is started.
        let (successor, reward) = env.step(state, &0, &mut rng, &mut ());
        assert_eq!(reward, Reward(0.0));
        let state = successor.into_continue().unwrap();
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: Some(()),
                prev_step: None,
                episode_done: false
            }
        );

        // Trial 0; Ep 1; Step 0
        // Take action 1 and get 0 reward
        // Inner state is terminal
        let (successor, feedback) = env.step(state, &1, &mut rng, &mut ());
        assert_eq!(feedback, Reward(0.0));
        let state = successor.into_continue().unwrap();
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: None,
                prev_step: Some(InnerStepObs {
                    action: 1,
                    feedback: Reward(0.0)
                }),
                episode_done: true
            }
        );

        // Trial 0; Ep 2; Init.
        // The action is ignored and a new inner episode is started.
        let (successor, feedback) = env.step(state, &1, &mut rng, &mut ());
        assert_eq!(feedback, Reward(0.0));
        let state = successor.into_continue().unwrap();
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: Some(()),
                prev_step: None,
                episode_done: false
            }
        );

        // Trial 0; Ep 2; Step 0
        // Take action 0 and get 1 reward
        // The inner state is terminal.
        // This inner episode was the last in the trial so the trial is done.
        let (successor, feedback) = env.step(state, &0, &mut rng, &mut ());
        assert_eq!(feedback, Reward(1.0));
        let state = successor.into_interrupt().unwrap();
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: None,
                prev_step: Some(InnerStepObs {
                    action: 0,
                    feedback: Reward(1.0)
                }),
                episode_done: true
            }
        );

        // Trial 1; Ep 1; Init.
        let state = env.initial_state(&mut rng);
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: Some(()),
                prev_step: None,
                episode_done: false
            }
        );

        // Trial 1; Ep 0; Step 0
        // Take action 0 and get 0 reward, since now 1 is the target action
        // Inner state is terminal.
        let (successor, reward) = env.step(state, &0, &mut rng, &mut ());
        assert_eq!(reward, Reward(0.0));
        let state = successor.into_continue().unwrap();
        assert_eq!(
            env.observe(&state, &mut rng),
            MetaObservation {
                inner_observation: None,
                prev_step: Some(InnerStepObs {
                    action: 0,
                    feedback: Reward(0.0)
                }),
                episode_done: true
            }
        );
    }
}
