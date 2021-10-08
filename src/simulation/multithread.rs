use super::hooks::BuildSimulationHook;
use super::{run_agent, Simulator, SimulatorError};
use crate::agents::{Agent, BuildManagerAgent, ManagerAgent};
use crate::envs::BuildEnv;
use crate::logging::TimeSeriesLogger;
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

    pub fn build_boxed_simulator<EC, MAC, HC>(
        &self,
        env_config: EC,
        manager_agent_config: MAC,
        hook_config: HC,
    ) -> Box<dyn Simulator>
    where
        EC: BuildEnv + 'static,
        EC::Environment: Send + 'static,
        EC::Observation: Clone,
        MAC: BuildManagerAgent<EC::ObservationSpace, EC::ActionSpace> + 'static,
        HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace> + 'static,
        HC::Hook: Send + 'static,
    {
        Box::new(MultithreadSimulator {
            env_config,
            manager_agent_config,
            num_workers: self.num_workers,
            hook_config,
        })
    }
}

/// Multithread simulator
pub struct MultithreadSimulator<EC, MAC, HC> {
    env_config: EC,
    manager_agent_config: MAC,
    num_workers: usize,
    hook_config: HC,
}

impl<EC, MAC, HC> Simulator for MultithreadSimulator<EC, MAC, HC>
where
    EC: BuildEnv,
    EC::Environment: Send + 'static,
    EC::Observation: Clone,
    MAC: BuildManagerAgent<EC::ObservationSpace, EC::ActionSpace>,
    HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace>,
    HC::Hook: Send + 'static,
{
    fn run_simulation(
        &mut self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), SimulatorError> {
        let env_structure = self.env_config.build_env(env_seed)?;
        let mut manager_agent = self
            .manager_agent_config
            .build_manager_agent(&env_structure, agent_seed)?;
        drop(env_structure);
        run_agent_multithread(
            &self.env_config,
            &mut manager_agent,
            self.num_workers,
            &self.hook_config,
            env_seed,
            agent_seed,
            logger,
        );
        Ok(())
    }
}

pub fn run_agent_multithread<EC, MA, HC>(
    env_config: &EC,
    agent_manager: &mut MA,
    num_workers: usize,
    worker_hook_config: &HC,
    env_seed: u64,
    agent_seed: u64,
    logger: &mut dyn TimeSeriesLogger,
) where
    EC: BuildEnv + ?Sized,
    EC::Environment: Send + 'static,
    MA: ManagerAgent,
    MA::Worker: Agent<EC::Observation, EC::Action> + 'static,
    EC::Observation: Clone,
    HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace>,
    HC::Hook: Send + 'static,
{
    let mut worker_threads = vec![];
    for i in 0..num_workers {
        let env_seed_i = env_seed.wrapping_add(i as u64);
        let mut env = env_config
            .build_env(env_seed_i)
            .expect("failed to build environment");
        let agent_seed_i = agent_seed.wrapping_add(i as u64);
        let mut worker = agent_manager.make_worker(agent_seed_i);
        let mut hook = worker_hook_config.build_hook(&env, num_workers, i);
        worker_threads.push(thread::spawn(move || {
            run_agent(&mut env, &mut worker, &mut hook, &mut ());
        }));
    }

    agent_manager.run(logger);
    for thread in worker_threads.into_iter() {
        thread.join().unwrap();
    }
}
