//! Simulating agent-environment interaction
pub mod hooks;
mod iter;

pub use hooks::{BuildSimulationHook, SimulationHook};
pub use iter::ActorSteps;

use crate::agents::{
    Actor, BatchUpdate, BuildAgentError, MakeActor, SynchronousUpdate, WriteHistoryBuffer,
};
use crate::envs::{BuildEnv, BuildEnvError, Environment, Successor};
use crate::logging::TimeSeriesLogger;
use rand::{rngs::StdRng, Rng, SeedableRng};
use thiserror::Error;

/// Runs agent-environment simulations.
pub trait Simulator {
    /// Run a simulation
    ///
    /// # Args
    /// * `env_seed` - Random seed used to derive the environment initialization seed(s).
    /// * `agent_seed` - Random seed used to derive the agent initialization seed(s).
    /// * `logger` - The logger for the main thread.
    fn run_simulation(
        &self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), SimulatorError>;
}

/// Error initializing or running a simulation.
#[derive(Error, Debug)]
pub enum SimulatorError {
    #[error("error building agent")]
    BuildAgent(#[from] BuildAgentError),
    #[error("error building environment")]
    BuildEnv(#[from] BuildEnvError),
}

/// Description of an environment step.
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

/// Run an agent-environment simulation.
///
/// Note that `Environment`, `SynchronousUpdate`, etc. are also implemented for mutable references
/// so this function can be called either with owned objects or with references.
///
/// # Args
/// * `environment` - The environment to simulate.
/// * `agent` - The agent to simulate.
/// * `hook` - A simulation hook run on each step. Controls when the simulation stops.
/// * `logger` - The logger to use.
pub fn run_agent<E, A, H, L>(environment: E, agent: A, mut hook: H, mut logger: L) -> A
where
    E: Environment,
    A: Actor<E::Observation, E::Action> + SynchronousUpdate<E::Observation, E::Action>,
    H: SimulationHook<E::Observation, E::Action>,
    L: TimeSeriesLogger,
{
    if !hook.start(&mut logger) {
        return agent;
    }
    let mut sim = environment.run(agent, logger);
    while sim.step_with(|_, agent, step, logger| {
        let continue_ = hook.call(&step, logger);
        agent.update(step, logger);
        continue_
    }) {}
    sim.actor
}

/// Train a batch agent in parallel.
pub fn train_parallel<T, EC>(
    agent: &mut T,
    env_config: &EC,
    num_epochs: usize,
    num_threads: usize,
    seed: u64,
    logger: &mut dyn TimeSeriesLogger,
) -> Result<(), SimulatorError>
where
    EC: BuildEnv + ?Sized,
    EC::Environment: Send,
    T: BatchUpdate<EC::Observation, EC::Action>,
    for<'a> T: MakeActor<'a, EC::Observation, EC::Action>,
    T::HistoryBuffer: Send,
{
    let mut seed_rng = StdRng::seed_from_u64(seed);
    let mut buffers: Vec<_> = (0..num_threads).map(|_| agent.new_buffer()).collect();

    for _ in 0..num_epochs {
        crossbeam::scope(|scope| -> Result<(), SimulatorError> {
            // Send a buffer to each thread to be filled
            let mut threads = Vec::new();
            for mut buffer in buffers.drain(..) {
                let env = env_config.build_env(seed_rng.gen())?;
                let actor = agent.make_actor(seed_rng.gen());
                threads.push(scope.spawn(move |_scope| {
                    let full = buffer.extend(env.run(actor, ()));
                    assert!(full);
                    buffer
                }));
            }

            buffers.extend(threads.into_iter().map(|t| t.join().unwrap()));
            Ok(())
        })
        .unwrap()?;

        agent.batch_update(&mut buffers, logger);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{
        buffers::SerialBufferConfig, testing, BatchedUpdatesConfig, BuildBatchAgent, PureAsActor,
        TabularQLearningAgentConfig,
    };
    use crate::envs::DeterministicBandit;

    #[test]
    fn train_parallel_tabular_q_bandit() {
        let env_config = DeterministicBandit::from_values(vec![0.0, 1.0]);
        let inner_agent_config = TabularQLearningAgentConfig::default();
        let history_buffer_config = SerialBufferConfig {
            soft_threshold: 100,
            hard_threshold: 110,
        };
        let agent_config = BatchedUpdatesConfig {
            agent_config: inner_agent_config,
            history_buffer_config,
        };
        let num_epochs = 10;
        let num_threads = 4;
        let seed = 0;
        let mut logger = ();

        let env = env_config.build_env(0).unwrap();
        let mut agent = agent_config.build_batch_agent(&env, 1).unwrap();
        train_parallel(
            &mut agent,
            &env_config,
            num_epochs,
            num_threads,
            seed,
            &mut logger,
        )
        .unwrap();

        testing::eval_deterministic_bandit(
            PureAsActor::new(agent, 1),
            &mut env_config.build_env(2).unwrap(),
            0.9,
        );
    }
}
