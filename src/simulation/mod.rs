//! Simulating agent-environment interaction
mod steps;
mod summary;
mod take_episodes;

pub use steps::{SimSeed, SimulatorSteps};
pub use summary::{OnlineStepsSummary, StepsSummary};
pub use take_episodes::TakeEpisodes;

use crate::agents::{ActorMode, Agent, BatchUpdate, WriteHistoryBuffer};
use crate::envs::{Environment, Successor};
use crate::logging::StatsLogger;
use crate::Prng;
use rand::SeedableRng;
use std::time::Instant;

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

/// Trait for simulation step iterators.
pub trait StepsIter<O, A>: Iterator<Item = PartialStep<O, A>> {
    /// Creates an iterator that yields steps from the first `n` episodes.
    #[inline]
    fn take_episodes(self, n: usize) -> TakeEpisodes<Self>
    where
        Self: Sized,
    {
        TakeEpisodes::new(self, n)
    }
}

impl<T, O, A> StepsIter<O, A> for T where T: Iterator<Item = PartialStep<O, A>> {}

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
        agent.batch_update_single(&mut buffer, logger);
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
        agent.batch_update_single(&mut buffer, logger);
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
///
/// The logger is used by the main thread for agent updates
/// as well as by one of the worker threads for action and step logs.
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
        let collect_start = Instant::now();

        // Send the logger to the first thread
        let mut worker0_logger = logger.with_scope("worker0");
        let mut send_logger = Some(&mut worker0_logger as &mut dyn StatsLogger);

        let summary = crossbeam::scope(|scope| {
            let mut threads = Vec::new();

            for (buffer, rngs) in buffers.iter_mut().zip(&mut thread_rngs) {
                let actor = agent.actor(ActorMode::Training);
                let thread_logger = send_logger.take();
                threads.push(scope.spawn(move |_scope| {
                    let mut summary = OnlineStepsSummary::default();
                    let ready = buffer.extend_until_ready(
                        SimulatorSteps::new(
                            environment,
                            actor,
                            &mut rngs.0,
                            &mut rngs.1,
                            thread_logger.unwrap_or(&mut ()),
                        )
                        .with_step_logging()
                        .map(|step| {
                            summary.push(&step);
                            step
                        }),
                    );
                    assert!(ready);
                    StepsSummary::from(summary)
                }));
            }

            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .sum::<StepsSummary>()
        })
        .unwrap();

        let mut sim_logger = logger.with_scope("sim");
        let mut episode_logger = (&mut sim_logger).with_scope("ep");
        let num_episodes = summary.episode_length.count();
        if num_episodes > 0 {
            episode_logger.log_scalar("reward_mean", summary.episode_reward.mean().unwrap());
            episode_logger.log_scalar("reward_stddev", summary.episode_reward.stddev().unwrap());
            episode_logger.log_scalar("length_mean", summary.episode_length.mean().unwrap());
            episode_logger.log_scalar("length_stddev", summary.episode_length.stddev().unwrap());
        }
        episode_logger.log_counter_increment("count", num_episodes);
        let mut step_logger = (&mut sim_logger).with_scope("step");
        let num_steps = summary.step_reward.count();
        if num_steps > 0 {
            step_logger.log_scalar("reward_mean", summary.step_reward.mean().unwrap());
            step_logger.log_scalar("reward_stddev", summary.step_reward.stddev().unwrap());
        }
        step_logger.log_counter_increment("count", num_steps);
        let update_start = Instant::now();
        sim_logger.log_duration("time", update_start - collect_start);

        agent.batch_update_slice(&mut buffers, &mut *logger);

        let mut agent_logger = logger.with_scope("agent_update");
        agent_logger.log_duration("time", update_start.elapsed());
        agent_logger.log_counter_increment("count", 1);
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
