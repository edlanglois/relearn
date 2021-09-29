use super::hooks::SimulationHook;
use super::{run_agent, RunSimulation};
use crate::agents::{Agent, ManagerAgent};
use crate::envs::BuildEnv;
use crate::logging::TimeSeriesLogger;
use std::convert::TryFrom;
use std::sync::{Arc, RwLock};
use std::thread;

/// Configuration for [`MultiThreadSimulator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiThreadSimulatorConfig {
    pub num_workers: usize,
    // TODO: Add seed
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
        <EB as BuildEnv>::Environment: 'static,
        MA: ManagerAgent + 'static,
        <MA as ManagerAgent>::Worker:
            Agent<<EB as BuildEnv>::Observation, <EB as BuildEnv>::Action> + 'static,
        <EB as BuildEnv>::Observation: Clone,
        H: SimulationHook<<EB as BuildEnv>::Observation, <EB as BuildEnv>::Action>
            + Clone
            + Send
            + 'static,
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
    <MA as ManagerAgent>::Worker:
        Agent<<EB as BuildEnv>::Observation, <EB as BuildEnv>::Action> + 'static,
    <EB as BuildEnv>::Observation: Clone,
    H: SimulationHook<<EB as BuildEnv>::Observation, <EB as BuildEnv>::Action>
        + Clone
        + Send
        + 'static,
{
    fn run_simulation(&mut self, logger: &mut dyn TimeSeriesLogger) {
        run_agent_multithread(
            &self.env_builder,
            &mut self.manager_agent,
            self.num_workers,
            &self.worker_hook,
            logger,
        );
    }
}

pub fn run_agent_multithread<EB, MA, H>(
    env_config: &Arc<RwLock<EB>>,
    agent_manager: &mut MA,
    num_workers: usize,
    worker_hook: &H,
    logger: &mut dyn TimeSeriesLogger,
) where
    EB: BuildEnv + Send + Sync + 'static,
    MA: ManagerAgent,
    <MA as ManagerAgent>::Worker:
        Agent<<EB as BuildEnv>::Observation, <EB as BuildEnv>::Action> + 'static,
    <EB as BuildEnv>::Observation: Clone,
    H: SimulationHook<<EB as BuildEnv>::Observation, <EB as BuildEnv>::Action>
        + Clone
        + Send
        + 'static,
{
    let mut worker_threads = vec![];
    for i in 0..num_workers {
        // TODO: Allow setting a seed
        let env_seed = 2 * u64::try_from(i).unwrap();
        let env_config_ = Arc::clone(env_config);
        let mut worker = agent_manager.make_worker(env_seed + 1);
        let mut hook = worker_hook.clone();
        worker_threads.push(thread::spawn(move || {
            let mut env = (*env_config_.read().unwrap()).build_env(env_seed).unwrap();
            drop(env_config_);
            run_agent(&mut env, &mut worker, &mut hook, &mut ());
        }));
    }

    agent_manager.run(logger);
    for thread in worker_threads.into_iter() {
        thread.join().unwrap();
    }
}
