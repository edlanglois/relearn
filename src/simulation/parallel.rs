use super::hooks::BuildStructuredHook;
use super::{run_agent, Simulator, SimulatorError};
use crate::agents::{Agent, ManagerAgent};
use crate::envs::BuildEnv;
use crate::logging::TimeSeriesLogger;
use std::sync::{Arc, RwLock};
use std::thread;

/// Configuration for [`ParallelSimulator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParallelSimulatorConfig {
    pub num_workers: usize,
}

impl Default for ParallelSimulatorConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
        }
    }
}

impl ParallelSimulatorConfig {
    pub const fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }

    pub fn build_simulator<EB, MA, HC>(
        &self,
        env_config: EB,
        manager_agent: MA,
        hook_config: HC,
    ) -> Box<dyn Simulator>
    where
        EB: BuildEnv + Send + Sync + 'static,
        EB::Environment: 'static,
        MA: ManagerAgent + 'static,
        MA::Worker: Agent<EB::Observation, EB::Action> + 'static,
        EB::Observation: Clone,
        HC: BuildStructuredHook<EB::ObservationSpace, EB::ActionSpace> + 'static,
        HC::Hook: Send + 'static,
    {
        Box::new(ParallelSimulator {
            env_builder: Arc::new(RwLock::new(env_config)),
            manager_agent,
            num_workers: self.num_workers,
            hook_config,
        })
    }
}

/// Multi-thread simulator
pub struct ParallelSimulator<EB, MA, HC> {
    env_builder: Arc<RwLock<EB>>,
    manager_agent: MA,
    num_workers: usize,
    hook_config: HC,
}

impl<EB, MA, HC> Simulator for ParallelSimulator<EB, MA, HC>
where
    EB: BuildEnv + Send + Sync + 'static,
    MA: ManagerAgent,
    MA::Worker: Agent<EB::Observation, EB::Action> + 'static,
    EB::Observation: Clone,
    HC: BuildStructuredHook<EB::ObservationSpace, EB::ActionSpace>,
    HC::Hook: Send + 'static,
{
    fn run_simulation(
        &mut self,
        env_seed: u64,
        agent_seed: u64,
        logger: &mut dyn TimeSeriesLogger,
    ) -> Result<(), SimulatorError> {
        run_agent_multithread(
            &self.env_builder,
            &mut self.manager_agent,
            self.num_workers,
            &self.hook_config,
            env_seed,
            agent_seed,
            logger,
        );
        Ok(())
    }
}

pub fn run_agent_multithread<EB, MA, HC>(
    env_config: &Arc<RwLock<EB>>,
    agent_manager: &mut MA,
    num_workers: usize,
    worker_hook_config: &HC,
    env_seed: u64,
    agent_seed: u64,
    logger: &mut dyn TimeSeriesLogger,
) where
    EB: BuildEnv + Send + Sync + 'static,
    MA: ManagerAgent,
    MA::Worker: Agent<EB::Observation, EB::Action> + 'static,
    EB::Observation: Clone,
    HC: BuildStructuredHook<EB::ObservationSpace, EB::ActionSpace>,
    HC::Hook: Send + 'static,
{
    // TODO: Avoid the extra env creation.
    let env_structure = env_config.read().unwrap().build_env(0).unwrap();
    let mut worker_threads = vec![];
    for i in 0..num_workers {
        let env_seed_i = env_seed.wrapping_add(i as u64);
        let env_config_ = Arc::clone(env_config);
        let agent_seed_i = agent_seed.wrapping_add(i as u64);
        let mut worker = agent_manager.make_worker(agent_seed_i);
        let mut hook = worker_hook_config.build_hook(&env_structure, num_workers, i);
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
