use super::{PartialStep, TransientStep};
use crate::agents::Actor;
use crate::envs::{Environment, Successor};
use crate::logging::{Loggable, StatsLogger};
use crate::Prng;
use rand::{Rng, SeedableRng};
use std::borrow::BorrowMut;
use std::iter::FusedIterator;

/// Iterator of environment-actor steps.
///
/// Uses separate PRNGs for the environment and actor so that a change in one will not affect the
/// other.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
{
    env: E,
    actor: T,
    rng_env: R,
    rng_actor: R,
    logger: L,

    state: Option<EpisodeState<E::State, E::Observation, T::EpisodeState>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct EpisodeState<ES, O, TS> {
    env: ES,
    observation: O,
    actor: TS,
}

/// Seed for simulation pseudo-random state.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SimSeed {
    /// Use a random seed derived from system entropy
    Random,
    /// A root seed from which both the environment and agent random states are derived.
    Root(u64),
    /// Individual seeds for the environment and agent random states.
    Individual { env: u64, agent: u64 },
}

impl SimSeed {
    /// Derive random number generators from this seed.
    fn derive_rngs<R: Rng + SeedableRng>(self) -> (R, R) {
        match self {
            SimSeed::Random => (R::from_entropy(), R::from_entropy()),
            SimSeed::Root(seed) => {
                let mut env_rng = R::seed_from_u64(seed);
                // For arbitrary R, R::from_rng might produce a PRNG correlated with
                // the original. Use the slower R::seed_from_u64 instead to ensure independence.
                let agent_rng = R::seed_from_u64(env_rng.gen());
                (env_rng, agent_rng)
            }
            SimSeed::Individual { env, agent } => (R::seed_from_u64(env), R::seed_from_u64(agent)),
        }
    }
}

impl<E, T, R, L> SimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: Rng + SeedableRng,
{
    pub fn new_seeded(env: E, actor: T, seed: SimSeed, logger: L) -> Self {
        let (rng_env, rng_actor) = seed.derive_rngs();
        Self::new(env, actor, rng_env, rng_actor, logger)
    }
}

impl<E, T, R, L> SimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
{
    pub fn new(env: E, actor: T, rng_env: R, rng_actor: R, logger: L) -> Self {
        Self {
            env,
            actor,
            rng_env,
            rng_actor,
            logger,
            state: None,
        }
    }
}

impl<E, T, R, L> SimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
    /// Execute one environment step then evaluate a closure on the resulting state.
    ///
    /// Use this if you need to access (a reference to) the successor state
    /// when the episode continues (a [`TransientStep`]).
    /// Otherwise, use the `Iterator` interface that returns [`PartialStep`].
    pub fn step_with<F, U>(&mut self, f: F) -> U
    where
        F: FnOnce(&mut E, &mut T, TransientStep<E::Observation, E::Action>, &mut L) -> U,
    {
        // Extract the current environment and actor state.
        // If None then start a new episode.
        let (env_state, observation, mut actor_state) = match self.state.take() {
            Some(state) => (state.env, state.observation, state.actor),
            None => {
                let env_state = self.env.initial_state(self.rng_env.borrow_mut());
                let observation = self.env.observe(&env_state, self.rng_env.borrow_mut());
                let actor_state = self.actor.new_episode_state(self.rng_actor.borrow_mut());
                (env_state, observation, actor_state)
            }
        };
        // Take an action with the actor given the observation.
        let action = self
            .actor
            .act(&mut actor_state, &observation, self.rng_actor.borrow_mut());
        // Take an environment step using this action.
        let (successor, reward) = self.env.step(
            env_state,
            &action,
            self.rng_env.borrow_mut(),
            &mut self.logger,
        );

        // Store the next state and observation if the environment continues.
        // Collect the successor observation information (converting Continue into a reference)
        // for passing to the callback function.
        assert!(self.state.is_none());
        let next = match successor {
            // Get the next state from `successor`.
            // Generate an observation and store both (along with actor_state) in `self.state`
            // so that we can use it on the next iteration.
            // Return a `Successor::Continue` that references the stored observation.
            Successor::Continue(next_state) => {
                let state_ref = self.state.insert(EpisodeState {
                    observation: self.env.observe(&next_state, self.rng_env.borrow_mut()),
                    env: next_state,
                    actor: actor_state,
                });
                Successor::Continue(&state_ref.observation)
            }
            Successor::Terminate => Successor::Terminate,
            // If the environment is interrupted then we do not need to store the next state
            // for future iterations. Return a Successor::Interrupt that owns the next state.
            Successor::Interrupt(next_state) => {
                Successor::Interrupt(self.env.observe(&next_state, self.rng_env.borrow_mut()))
            }
        };

        let step = TransientStep {
            observation,
            action,
            reward,
            next,
        };

        f(&mut self.env, &mut self.actor, step, &mut self.logger)
    }

    pub fn with_step_logging(self) -> LoggedSimulatorSteps<E, T, R, L> {
        LoggedSimulatorSteps::new(self)
    }
}

impl<E, T, R, L> Iterator for SimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
    /// Cannot return a `TransientStep` without Generic Associated Types
    type Item = PartialStep<E::Observation, E::Action>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step_with(|_, _, s, _| s.into_partial()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // infinite
        (usize::MAX, None)
    }
}

impl<E, A, R, L> FusedIterator for SimulatorSteps<E, A, R, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
}

/// Simulator steps with logging
pub struct LoggedSimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
{
    simulator: SimulatorSteps<E, T, R, L>,
    episode_reward: f64,
    episode_length: u64,
}

impl<E, T, R, L> LoggedSimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
{
    pub fn new(simulator: SimulatorSteps<E, T, R, L>) -> Self {
        Self {
            simulator,
            episode_reward: 0.0,
            episode_length: 0,
        }
    }
}

impl<E, T, R, L> Iterator for LoggedSimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
    /// Cannot return a `TransientStep` without Generic Associated Types
    type Item = PartialStep<E::Observation, E::Action>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.simulator.step_with(|_, _, step, logger| {
            // Check for flushing only on the first log of each episode boundary
            logger.log_scalar("step/reward", step.reward);
            // TODO: Log action and observation
            logger
                .log_no_flush("step/count".into(), Loggable::CounterIncrement(1))
                .unwrap();
            self.episode_reward += step.reward;
            self.episode_length += 1;
            if step.next.episode_done() {
                logger
                    .log_no_flush(
                        "episode/reward".into(),
                        Loggable::Scalar(self.episode_reward),
                    )
                    .unwrap();
                logger
                    .log_no_flush(
                        "episode/length".into(),
                        Loggable::Scalar(self.episode_length as f64),
                    )
                    .unwrap();
                logger
                    .log_no_flush("episode/count".into(), Loggable::CounterIncrement(1))
                    .unwrap();
                self.episode_reward = 0.0;
                self.episode_length = 0;
            }

            step.into_partial()
        }))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // infinite
        (usize::MAX, None)
    }
}
