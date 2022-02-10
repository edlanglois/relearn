//! Simulating agent-environment interaction
mod steps;

pub use steps::{SimulationSummary, SimulatorSteps};

use crate::agents::{ActorMode, Agent, BatchUpdate, WriteHistoryBuffer};
use crate::envs::{Environment, Successor};
use crate::logging::StatsLogger;
use crate::Prng;
use rand::SeedableRng;
use std::iter;

/// Description of an environment step.
///
/// There are a few different forms that this structure can take in terms of describing the
/// next observation when `next` is [`Successor::Continue`].
/// These are determined by the value of the third generic parameter `U`:
/// * `Step<O, A>` - `U = O` - The continuing successor observation is owned.
/// * [`TransientStep<O, A>`] - `U = &O` - The continuing successor observation is borrowed.
/// * [`PartialStep<O, A>`] - `U = ()` - The continuing successor observation is omitted.
///
/// If `next` is [`Successor::Interrupt`] then the observation is owned in all cases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step<O, A, U = O> {
    /// The initial observation.
    pub observation: O,
    /// The action taken from the initial state given the initial observation.
    pub action: A,
    /// The resulting reward.
    pub reward: f64,
    /// The next observation or outcome; how the episode progresses.
    pub next: Successor<O, U>,
}

impl<O, A, U> Step<O, A, U> {
    pub const fn new(observation: O, action: A, reward: f64, next: Successor<O, U>) -> Self {
        Self {
            observation,
            action,
            reward,
            next,
        }
    }

    pub fn into_partial(self) -> PartialStep<O, A> {
        Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: self.next.into_partial(),
        }
    }
}

/// Description of an environment step where the successor observation is borrowed.
pub type TransientStep<'a, O, A> = Step<O, A, &'a O>;

impl<O: Clone, A> TransientStep<'_, O, A> {
    /// Convert a transient step into an owned step by cloning any borrowed successor observation.
    #[inline]
    pub fn into_owned(self) -> Step<O, A> {
        Step {
            observation: self.observation,
            action: self.action,
            reward: self.reward,
            next: self.next.into_owned(),
        }
    }
}

/// Partial description of an environment step.
///
/// The successor state is omitted when the episode continues.
/// Using this can help avoid copying the observation.
pub type PartialStep<O, A> = Step<O, A, ()>;

/// Train a batch learning agent in this thread.
pub fn train_serial<T, E>(
    agent: &mut T,
    environment: &E,
    num_periods: usize,
    rng_env: &mut Prng,
    rng_agent: &mut Prng,
    logger: &mut dyn StatsLogger,
) where
    T: Agent<E::Observation, E::Action> + ?Sized,
    E: Environment + ?Sized,
{
    let mut buffer = agent.buffer(agent.batch_size_hint());
    for _ in 0..num_periods {
        let ready = buffer.extend_until_ready(
            SimulatorSteps::new(
                environment,
                agent.actor(ActorMode::Training),
                &mut *rng_env,
                &mut *rng_agent,
                &mut *logger,
            )
            .with_step_logging(),
        );
        assert!(ready);
        agent.batch_update(iter::once(&mut buffer), logger);
    }
}

/// Train a batch learning agent in this thread with a callback function evaluated on each step.
pub fn train_serial_callback<T, E, F>(
    agent: &mut T,
    environment: &E,
    mut f: F,
    num_periods: usize,
    rng_env: &mut Prng,
    rng_agent: &mut Prng,
    logger: &mut dyn StatsLogger,
) where
    T: Agent<E::Observation, E::Action> + ?Sized,
    E: Environment + ?Sized,
    F: FnMut(&PartialStep<E::Observation, E::Action>),
{
    let mut buffer = agent.buffer(agent.batch_size_hint());
    for _ in 0..num_periods {
        let ready = buffer.extend_until_ready(
            SimulatorSteps::new(
                environment,
                agent.actor(ActorMode::Training),
                &mut *rng_env,
                &mut *rng_agent,
                &mut *logger,
            )
            .with_step_logging()
            .map(|step| {
                f(&step);
                step
            }),
        );
        assert!(ready);
        agent.batch_update(iter::once(&mut buffer), logger);
    }
}

/// Configuration for [`train_parallel`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TrainParallelConfig {
    /// Number of data-collection-batch-update loops.
    pub num_periods: usize,
    /// Number of simulation threads.
    pub num_threads: usize,
    /// Minimum step capacity of each worker buffer.
    pub min_workers_steps: usize,
}

/// Train a batch learning agent in parallel across several threads.
pub fn train_parallel<T, E>(
    agent: &mut T,
    environment: &E,
    config: &TrainParallelConfig,
    rng_env: &mut Prng,
    rng_agent: &mut Prng,
    logger: &mut dyn StatsLogger,
) where
    E: Environment + Sync + ?Sized,
    T: Agent<E::Observation, E::Action> + ?Sized,
    T::Actor: Send,
    T::HistoryBuffer: Send,
{
    let capacity = agent
        .batch_size_hint()
        .divide(config.num_threads)
        .with_steps_at_least(config.min_workers_steps);
    let mut buffers: Vec<_> = (0..config.num_threads)
        .map(|_| agent.buffer(capacity))
        .collect();
    let mut thread_rngs: Vec<_> = (0..config.num_threads)
        .map(|_| {
            (
                Prng::from_rng(&mut *rng_env).expect("Prng should be infallible"),
                Prng::from_rng(&mut *rng_agent).expect("Prng should be infallible"),
            )
        })
        .collect();

    for _ in 0..config.num_periods {
        crossbeam::scope(|scope| {
            let mut threads = Vec::new();
            // Send the logger to the first thread
            let mut send_logger = Some(&mut *logger);
            for (buffer, rngs) in buffers.iter_mut().zip(&mut thread_rngs) {
                let actor = agent.actor(ActorMode::Training);
                let thread_logger = send_logger.take();
                threads.push(scope.spawn(move |_scope| {
                    let ready = buffer.extend_until_ready(
                        SimulatorSteps::new(
                            environment,
                            actor,
                            &mut rngs.0,
                            &mut rngs.1,
                            thread_logger.unwrap_or(&mut ()),
                        )
                        .with_step_logging(),
                    );
                    assert!(ready);
                }));
            }

            for thread in threads {
                thread.join().unwrap()
            }
        })
        .unwrap();

        agent.batch_update(&mut buffers, logger);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{testing, BuildAgent, TabularQLearningAgentConfig};
    use crate::envs::DeterministicBandit;

    #[test]
    fn train_parallel_tabular_q_bandit() {
        let config = TrainParallelConfig {
            num_periods: 10,
            num_threads: 4,
            min_workers_steps: 100,
        };
        let mut rng_env = Prng::seed_from_u64(0);
        let mut rng_actor = Prng::seed_from_u64(1);
        let mut logger = ();

        let env = DeterministicBandit::from_values(vec![0.0, 1.0]);
        let mut agent = TabularQLearningAgentConfig::default()
            .build_agent(&env, &mut rng_actor)
            .unwrap();

        train_parallel(
            &mut agent,
            &env,
            &config,
            &mut rng_env,
            &mut rng_actor,
            &mut logger,
        );

        testing::eval_deterministic_bandit(agent.actor(ActorMode::Evaluation), &env, 0.9);
    }
}
