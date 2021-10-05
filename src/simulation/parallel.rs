use super::hooks::SimulationHook;
use super::{run_agent, RunSimulation};
use crate::agents::{Agent, ManagerAgent};
use crate::envs::BuildEnv;
use crate::error::RLError;
use crate::logging::TimeSeriesLogger;
use std::sync::{Arc, RwLock};
use std::thread;

/// Configuration for [`MultiThreadSimulator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiThreadSimulatorConfig {
    pub num_workers: usize,
}

impl Default for MultiThreadSimulatorConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
        }
    }
}

impl MultiThreadSimulatorConfig {
    pub const fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }

    pub fn build_simulator<EB, MA, H>(
        &self,
        env_config: EB,
        manager_agent: MA,
        worker_hook: H,
    ) -> Box<dyn RunSimulation>
    where
        EB: BuildEnv + Send + Sync + 'static,
        EB::Environment: 'static,
        MA: ManagerAgent + 'static,
        MA::Worker: Agent<EB::Observation, EB::Action> + 'static,
        EB::Observation: Clone,
        H: SimulationHook<EB::Observation, EB::Action> + Clone + Send + 'static,
    {
        Box::new(MultiThreadSimulator {
            env_builder: Arc::new(RwLock::new(env_config)),
            manager_agent,
            num_workers: self.num_workers,
            worker_hook,
        })
    }
}

/// Multi-thread simulator
pub struct MultiThreadSimulator<EB, MA, H> {
    env_builder: Arc<RwLock<EB>>,
    manager_agent: MA,
    num_workers: usize,
    worker_hook: H,
}

impl<EB, MA, H> RunSimulation for MultiThreadSimulator<EB, MA, H>
where
    EB: BuildEnv + Send + Sync + 'static,
    MA: ManagerAgent,
    MA::Worker: Agent<EB::Observation, EB::Action> + 'static,
    EB::Observation: Clone,
    H: SimulationHook<EB::Observation, EB::Action> + Clone + Send + 'static,
{
    fn run_simulation(
        &mut self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), RLError> {
        run_agent_multithread(
            &self.env_builder,
            &mut self.manager_agent,
            self.num_workers,
            &self.worker_hook,
            env_seed,
            agent_seed,
            logger,
        );
        Ok(())
    }
}

pub fn run_agent_multithread<EB, MA, H>(
    env_config: &Arc<RwLock<EB>>,
    agent_manager: &mut MA,
    num_workers: usize,
    worker_hook: &H,
    env_seed: u64,
    agent_seed: u64,
    logger: &mut dyn TimeSeriesLogger,
) where
    EB: BuildEnv + Send + Sync + 'static,
    MA: ManagerAgent,
    MA::Worker: Agent<EB::Observation, EB::Action> + 'static,
    EB::Observation: Clone,
    H: SimulationHook<EB::Observation, EB::Action> + Clone + Send + 'static,
{
    let mut worker_threads = vec![];
    for i in 0..num_workers {
        let env_seed_i = env_seed.wrapping_add(i as u64);
        let env_config_ = Arc::clone(env_config);
        let agent_seed_i = agent_seed.wrapping_add(i as u64);
        let mut worker = agent_manager.make_worker(agent_seed_i);
        let mut hook = worker_hook.clone();
        worker_threads.push(thread::spawn(move || {
            let mut env = (*env_config_.read().unwrap())
                .build_env(env_seed_i)
                .unwrap();
            drop(env_config_);
            run_agent(&mut env, &mut worker, &mut hook, &mut ());
        }));
    }

    agent_manager.run(logger);
    for thread in worker_threads.into_iter() {
        thread.join().unwrap();
    }
}
