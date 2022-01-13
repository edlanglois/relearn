use super::{PartialStep, TransientStep};
use crate::agents::Actor;
use crate::envs::{Environment, Successor};
use crate::logging::{Event, TimeSeriesLogger};
use crate::Prng;
use std::borrow::BorrowMut;
use std::fmt;
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

impl<E, A, R, L> SimulatorSteps<E, A, R, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: TimeSeriesLogger,
{
    /// Execute one environment step then evaluate a closure on the resulting state.
    pub fn step_with<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut E, &mut A, TransientStep<E::Observation, E::Action>, &mut L) -> T,
    {
        let (env_state, observation, mut actor_state) = match self.state.take() {
            Some(state) => (state.env, state.observation, state.actor),
            None => {
                let env_state = self.env.initial_state(self.rng_env.borrow_mut());
                let observation = self.env.observe(&env_state, self.rng_env.borrow_mut());
                let actor_state = self.actor.new_episode_state(self.rng_actor.borrow_mut());
                (env_state, observation, actor_state)
            }
        };
        let action = self
            .actor
            .act(&mut actor_state, &observation, self.rng_actor.borrow_mut());
        let (successor, reward) = self.env.step(
            env_state,
            &action,
            self.rng_env.borrow_mut(),
            &mut self.logger.event_logger(Event::EnvStep),
        );

        // Store the next state and observation if the environment continues.
        // Collect the successor observation information (converting Continue into a reference)
        // for passing to the callback function.
        assert!(self.state.is_none());
        let next = match successor {
            Successor::Continue(next_state) => {
                let state_ref = self.state.insert(EpisodeState {
                    observation: self.env.observe(&next_state, self.rng_env.borrow_mut()),
                    env: next_state,
                    actor: actor_state,
                });
                Successor::Continue(&state_ref.observation)
            }
            Successor::Terminate => Successor::Terminate,
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
}

impl<E, A, R, L> Iterator for SimulatorSteps<E, A, R, L>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: TimeSeriesLogger,
{
    /// Cannot return a `TransientStep` without Generic Associated Types
    type Item = PartialStep<E::Observation, E::Action>;

    #[inline]
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
    L: TimeSeriesLogger,
{
}

/* XXX
/// An iterator of mapped environment-actor steps.
///
/// Created by `SimulatorSteps::map_steps`.
/// This allows mapping transient steps, while `<SimulatorSteps as Iterator>::map` maps partial
/// steps.
pub struct SimulatorMappedSteps<E, A, R, L, F> {
    simulator: SimulatorSteps<E, A, R, L>,
    f: F,
}

impl<E, A, R, L, F, U> Iterator for SimulatorMappedSteps<E, A, R, L, F>
where
    E: Environment,
    A: Actor<E::Observation, E::Action>,
    R: BorrowMut<Prng>,
    L: TimeSeriesLogger,
    F: FnMut(&TransientStep<E::Observation, E::Action>) -> U,
{
    type Item = U;
}
*/

/* XXX
pub struct TrainingSimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: BatchUpdate<E::Observation, E::Action>,
{
    steps: SimulatorSteps<E, T, R, L>,
    buffer: T::HistoryBuffer,
}
*/

/*
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrainingSimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Agent<E::Observation, E::Action>,
{
    steps: SimulatorSteps<E, SerialActorAgent<T, E::Observation, E::Action>, R, L>,
}

impl<E, T, R, L> TrainingSimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Agent<E::Observation, E::Action>,
{
    pub fn new(env: E, agent: T, rng_env: R, rng_actor: R, logger: L) -> Self {
        Self {
            steps: SimulatorSteps::new(
                env,
                SerialActorAgent::new(agent),
                rng_env,
                rng_actor,
                logger,
            ),
        }
    }

    /// Execute one environment step then evaluate a closure on the resulting state.
    pub fn step_with<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut E, &mut A, TransientStep<E::Observation, E::Action>, &mut L) -> T,

}

impl<E, T, R, L> Iterator for TrainingSimulatorSteps<E, T, R, L>
where
    E: Environment,
    T: Agent<E::Observation, E::Action>,
{
    type Item = PartialStep<E::Observation, E::Action>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step_with(|_, agent, step, logger| {
            let step = step.into_partial();
            agent.update(step.into_partial(), logger)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // infinite
        (usize::MAX, None)
    }
}
*/

/// Basic summary statistics of a simulation
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct SimulationSummary {
    pub num_steps: u64,
    pub num_episodes: u64,
    pub total_reward: f64,
}
impl fmt::Display for SimulationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "num_steps: {}", self.num_steps)?;
        writeln!(f, "num_episodes: {}", self.num_episodes)?;
        writeln!(
            f,
            "mean_step_reward: {}",
            self.total_reward / self.num_steps as f64
        )?;
        writeln!(
            f,
            "mean_ep_reward:   {}",
            self.total_reward / self.num_episodes as f64
        )?;
        writeln!(
            f,
            "mean_ep_length:   {}",
            self.num_steps as f64 / self.num_episodes as f64
        )?;
        Ok(())
    }
}

// TODO: Make this a trait
impl SimulationSummary {
    pub fn update<O, A>(&mut self, step: &PartialStep<O, A>) {
        self.num_steps += 1;
        self.total_reward += step.reward;
        if step.next.episode_done() {
            self.num_episodes += 1;
        }
    }

    pub fn from_steps<I: Iterator<Item = PartialStep<O, A>>, O, A>(steps: I) -> Self {
        steps.fold(Self::default(), |mut s, step| {
            s.update(&step);
            s
        })
    }
}
