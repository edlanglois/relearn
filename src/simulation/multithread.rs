use super::hooks::BuildSimulationHook;
use super::{run_agent, Simulator, SimulatorError};
use crate::agents::{BuildMultithreadAgent, InitializeMultithreadAgent, MultithreadAgentManager};
use crate::envs::{BuildEnv, EnvStructure};
use crate::logging::{BuildThreadLogger, TimeSeriesLogger};
use std::thread;

/// Configuration for [`MultithreadSimulator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultithreadSimulatorConfig {
    pub num_workers: usize,
}

impl Default for MultithreadSimulatorConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
        }
    }
}

impl MultithreadSimulatorConfig {
    pub const fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }

    pub const fn build_simulator<EC, AC, HC, LC>(
        &self,
        env_config: EC,
        agent_config: AC,
        hook_config: HC,
        worker_logger_config: LC,
    ) -> MultithreadSimulator<EC, AC, HC, LC> {
        MultithreadSimulator {
            env_config,
            agent_config,
            hook_config,
            worker_logger_config,
            num_workers: self.num_workers,
        }
    }
}

/// An agent-environment simulator with multiple individual simulation threads.
#[derive(Debug)]
pub struct MultithreadSimulator<EC, AC, HC, LC> {
    pub env_config: EC,
    pub agent_config: AC,
    pub hook_config: HC,
    pub worker_logger_config: LC,
    pub num_workers: usize,
}

impl<EC, AC, HC, LC> MultithreadSimulator<EC, AC, HC, LC>
where
    EC: BuildEnv,
    EC::Observation: Clone,
    EC::Environment: Send + 'static,
    AC: BuildMultithreadAgent<EC::ObservationSpace, EC::ActionSpace>,
    HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace>,
    HC::Hook: Send + 'static,
    LC: BuildThreadLogger,
    LC::ThreadLogger: TimeSeriesLogger + Send + 'static,
{
    #[allow(clippy::type_complexity)]
    /// Run a simulation and return the trained manager agent.
    pub fn train(
        &self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<
        <AC::MultithreadAgent as InitializeMultithreadAgent<EC::Observation, EC::Action>>::Manager,
        SimulatorError,
    > {
        let mut worker_threads = vec![];
        let env_structure: &dyn EnvStructure<ObservationSpace = _, ActionSpace = _> =
            &self.env_config.build_env(env_seed).unwrap();
        let agent_initializer = self
            .agent_config
            .build_multithread_agent(env_structure, agent_seed)
            .unwrap();

        for i in 0..self.num_workers {
            let mut env = self.env_config.build_env(env_seed.wrapping_add(i as u64))?;
            let mut worker = agent_initializer.make_worker(i);
            let mut hook = self.hook_config.build_hook(&env, self.num_workers, i);
            let mut worker_logger = self.worker_logger_config.build_thread_logger(i);
            worker_threads.push(thread::spawn(move || {
                run_agent(&mut env, &mut worker, &mut hook, &mut worker_logger);
            }));
        }

        let mut manager = agent_initializer.into_manager();
        manager.run(logger);

        for thread in worker_threads.into_iter() {
            thread.join().expect("Worker thread panic");
        }
        Ok(manager)
    }
}

impl<EC, AC, HC, LC> Simulator for MultithreadSimulator<EC, AC, HC, LC>
where
    EC: BuildEnv,
    EC::Observation: Clone,
    EC::Environment: Send + 'static,
    AC: BuildMultithreadAgent<EC::ObservationSpace, EC::ActionSpace>,
    HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace>,
    HC::Hook: Send + 'static,
    LC: BuildThreadLogger,
    LC::ThreadLogger: TimeSeriesLogger + Send + 'static,
{
    fn run_simulation(
        &self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), SimulatorError> {
        self.train(env_seed, agent_seed, logger)?;
        Ok(())
    }
}
