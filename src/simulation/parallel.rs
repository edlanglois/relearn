use super::hooks::SimulationHook;
use super::{run_agent, BuildSimError, RunSimulation, SimulatorBuilder};
use crate::agents::{Agent, ManagerAgent};
use crate::envs::{EnvBuilder, EnvStructure, StatefulEnvironment};
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;
use std::convert::TryFrom;
use std::marker::PhantomData;
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
}

impl<EB, E, A, H> SimulatorBuilder<MultiThreadSimulator<EB, E, A, H>, EB, E, A, H>
    for MultiThreadSimulatorConfig
where
    EB: EnvBuilder<E>,
{
    fn build_simulator(
        &self,
        env_config: EB,
        agent: A,
        hook: H,
    ) -> Result<MultiThreadSimulator<EB, E, A, H>, BuildSimError> {
        Ok(MultiThreadSimulator {
            env_builder: env_config,
            env_type: PhantomData,
            manager_agent: agent,
            num_workers: self.num_workers,
            worker_hook: hook,
        })
    }
}

/// Multi-thread simulator
pub struct MultiThreadSimulator<EB, E, MA, H> {
    env_builder: EB,
    // *const E to avoid indicating ownership. See:
    // https://doc.rust-lang.org/std/marker/struct.PhantomData.html#ownership-and-the-drop-check
    env_type: PhantomData<*const E>,
    manager_agent: MA,
    num_workers: usize,
    worker_hook: H,
}

impl<EB, E, MA, H> RunSimulation for MultiThreadSimulator<EB, E, MA, H>
where
    EB: EnvBuilder<E>,
    E: StatefulEnvironment + Send + 'static,
    MA: ManagerAgent,
    <MA as ManagerAgent>::Worker: Agent<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + 'static,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    H: SimulationHook<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + Clone
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

pub fn run_agent_multithread<EB, E, MA, H>(
    env_builder: &EB,
    manager: &mut MA,
    num_workers: usize,
    worker_hook: &H,
    logger: &mut dyn TimeSeriesLogger,
) where
    EB: EnvBuilder<E>,
    E: StatefulEnvironment + Send + 'static,
    MA: ManagerAgent,
    <MA as ManagerAgent>::Worker: Agent<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + 'static,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    H: SimulationHook<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + Clone
        + Send
        + 'static,
{
    let mut worker_threads = vec![];
    for i in 0..num_workers {
        // TODO: Allow setting a seed
        let env_seed = 2 * u64::try_from(i).unwrap();
        let mut env: E = env_builder.build_env(env_seed).unwrap();
        let mut worker = manager.make_worker(env_seed + 1);
        let mut hook = worker_hook.clone();
        worker_threads.push(thread::spawn(move || {
            run_agent(&mut env, &mut worker, &mut hook, &mut ());
        }));
    }

    manager.run(logger);
    for thread in worker_threads.into_iter() {
        thread.join().unwrap();
    }
}
