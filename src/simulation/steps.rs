use super::{PartialStep, SimSeed};
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
pub struct Steps<E, T, R, L>
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

impl<E, T, R, L> Steps<E, T, R, L>
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

impl<E, T, R, L> Steps<E, T, R, L>
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

impl<E, T, R, L> Steps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
    /// Advance to the next environment-actor step.
    pub fn step(&mut self) -> PartialStep<E::Observation, E::Action> {
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
        debug_assert!(self.state.is_none());
        let next = match successor {
            // Get the next state from `successor`.
            // Generate an observation and store both (along with actor_state) in `self.state`
            // so that we can use it on the next iteration.
            // Return a `Successor::Continue` that references the stored observation.
            Successor::Continue(next_state) => {
                self.state = Some(EpisodeState {
                    observation: self.env.observe(&next_state, self.rng_env.borrow_mut()),
                    env: next_state,
                    actor: actor_state,
                });
                // Note: Transient version would instead return Continue(&self.state.observation)
                Successor::Continue(())
            }
            Successor::Terminate => Successor::Terminate,
            // If the environment is interrupted then we do not need to store the next state
            // for future iterations. Return a Successor::Interrupt that owns the next state.
            Successor::Interrupt(next_state) => {
                Successor::Interrupt(self.env.observe(&next_state, self.rng_env.borrow_mut()))
            }
        };

        PartialStep {
            observation,
            action,
            reward,
            next,
        }
    }
}

impl<E, T, R, L> Steps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
    pub fn with_step_logging(self) -> LoggedSteps<E, T, R, L> {
        LoggedSteps::new(self)
    }
}

impl<E, T, R, L> Iterator for Steps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
    /// Cannot return a `TransientStep` without Generic Associated Types
    type Item = PartialStep<E::Observation, E::Action>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // infinite
        (usize::MAX, None)
    }
}

impl<E, A, R, L> FusedIterator for Steps<E, A, R, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
}

/// Simulator steps with logging
pub struct LoggedSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
{
    simulator: Steps<E, T, R, L>,
    episode_reward: f64,
    episode_length: u64,
}

impl<E, T, R, L> LoggedSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
{
    pub fn new(simulator: Steps<E, T, R, L>) -> Self {
        Self {
            simulator,
            episode_reward: 0.0,
            episode_length: 0,
        }
    }
}

impl<E, T, R, L> Iterator for LoggedSteps<E, T, R, L>
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: StatsLogger,
{
    /// Cannot return a `TransientStep` without Generic Associated Types
    type Item = PartialStep<E::Observation, E::Action>;

    fn next(&mut self) -> Option<Self::Item> {
        let step = self.simulator.next()?;

        let mut step_logger = (&mut self.simulator.logger).with_scope("step");
        step_logger.log_scalar("reward", step.reward);
        // TODO: Log action and observation
        step_logger
            .log_no_flush("count".into(), Loggable::CounterIncrement(1))
            .unwrap();
        self.episode_reward += step.reward;
        self.episode_length += 1;
        if step.next.episode_done() {
            let mut episode_logger = (&mut self.simulator.logger).with_scope("episode");
            episode_logger
                .log_no_flush("reward".into(), Loggable::Scalar(self.episode_reward))
                .unwrap();
            episode_logger
                .log_no_flush(
                    "length".into(),
                    Loggable::Scalar(self.episode_length as f64),
                )
                .unwrap();
            episode_logger
                .log_no_flush("count".into(), Loggable::CounterIncrement(1))
                .unwrap();
            self.episode_reward = 0.0;
            self.episode_length = 0;
        }

        Some(step)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // infinite
        (usize::MAX, None)
    }
}
