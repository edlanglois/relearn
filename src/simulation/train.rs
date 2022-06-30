use super::{OnlineStepsSummary, Simulation, Steps, StepsSummary};
use crate::agents::{buffers::HistoryDataBound, ActorMode, Agent, BatchUpdate, WriteExperience};
use crate::envs::{EnvStructure, Environment, StructuredEnvironment};
use crate::feedback::{Feedback, Summary};
use crate::logging::{Loggable, StatsLogger};
use crate::spaces::{LogElementSpace, Space};
use crate::Prng;
use log::warn;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::iter;
use std::time::Instant;

/// Train a batch learning agent in this thread.
pub fn train_serial<T, E>(
    agent: &mut T,
    environment: &E,
    num_periods: usize,
    rng_env: &mut Prng,
    rng_agent: &mut Prng,
    logger: &mut dyn StatsLogger,
) where
    T: Agent<E::Observation, E::Action>
        + BatchUpdate<E::Observation, E::Action, Feedback = E::Feedback>
        + ?Sized,
    E: StructuredEnvironment + ?Sized,
    E::ObservationSpace: LogElementSpace,
    E::ActionSpace: LogElementSpace,
    E::Feedback: Feedback,
{
    let mut buffer = agent.buffer();
    for _ in 0..num_periods {
        let update_size = agent.min_update_size();
        buffer
            .write_experience(
                update_size
                    .take(Steps::new(
                        environment,
                        agent.actor(ActorMode::Training),
                        &mut *rng_env,
                        &mut *rng_agent,
                        &mut *logger,
                    ))
                    .log(),
            )
            .unwrap_or_else(|err| warn!("error filling buffer: {}", err));
        agent.batch_update(iter::once(&mut buffer), logger);
    }
}

/// Configuration for [`train_parallel`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrainParallelConfig {
    /// Number of data-collection-batch-update loops.
    pub num_periods: usize,
    /// Number of simulation threads.
    pub num_threads: usize,
    /// Minimum step capacity of each worker buffer.
    pub min_worker_steps: usize,
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
    // TODO: Why does the simpler bound work for train_serial but not here?
    E: EnvStructure
        + Environment<
            Observation = <E::ObservationSpace as Space>::Element,
            Action = <E::ActionSpace as Space>::Element,
            Feedback = <E::FeedbackSpace as Space>::Element,
        > + Sync
        + ?Sized,
    T: Agent<<E::ObservationSpace as Space>::Element, <E::ActionSpace as Space>::Element>
        + BatchUpdate<
            <E::ObservationSpace as Space>::Element,
            <E::ActionSpace as Space>::Element,
            Feedback = <E::FeedbackSpace as Space>::Element,
        > + ?Sized,
    T::Actor: Send,
    T::HistoryBuffer: Send,
    E::ObservationSpace: LogElementSpace,
    E::ActionSpace: LogElementSpace,
    <E::FeedbackSpace as Space>::Element: Feedback,
    <<E::FeedbackSpace as Space>::Element as Feedback>::StepSummary: Send,
    <<E::FeedbackSpace as Space>::Element as Feedback>::EpisodeSummary: Send,
{
    let mut buffers: Vec<_> = (0..config.num_threads).map(|_| agent.buffer()).collect();
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

        let worker_update_size =
            agent
                .min_update_size()
                .divide(config.num_threads)
                .max(HistoryDataBound {
                    min_steps: config.min_worker_steps,
                    slack_steps: 0,
                });

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
                    buffer
                        .write_experience(
                            worker_update_size
                                .take(Steps::new(
                                    environment,
                                    actor,
                                    &mut rngs.0,
                                    &mut rngs.1,
                                    thread_logger.unwrap_or(&mut ()),
                                ))
                                .log()
                                .map(|step| {
                                    summary.push(&step);
                                    step
                                }),
                        )
                        .unwrap_or_else(|err| warn!("error filling buffer: {}", err));
                    StepsSummary::from(summary)
                }));
            }

            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .sum::<StepsSummary<_>>()
        })
        .unwrap();

        let mut sim_logger = logger.with_scope("sim").group();
        let mut episode_logger = (&mut sim_logger).with_scope("ep");
        let num_episodes = summary.episode_length.count();
        if num_episodes > 0 {
            summary
                .episode_feedback
                .log("fbk", &mut episode_logger)
                .unwrap();
            episode_logger.log_scalar("length_mean", summary.episode_length.mean().unwrap());
            episode_logger.log_scalar("length_stddev", summary.episode_length.stddev().unwrap());
        }
        episode_logger.log_counter_increment("count", num_episodes);

        let mut step_logger = (&mut sim_logger).with_scope("step");
        summary.step_feedback.log("fbk", &mut step_logger).unwrap();
        step_logger.log_counter_increment("count", summary.step_feedback.size());
        let update_start = Instant::now();
        sim_logger.log_duration("time", update_start - collect_start);
        drop(sim_logger);

        agent.batch_update(&mut buffers, &mut *logger);

        let mut agent_logger = logger.with_scope("agent_update").group();
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
            min_worker_steps: 100,
        };
        let mut rng_env = Prng::seed_from_u64(0);
        let mut rng_actor = Prng::seed_from_u64(1);

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
            &mut (),
        );

        testing::eval_deterministic_bandit(agent.actor(ActorMode::Evaluation), &env, 0.9);
    }
}
