use super::RunSimulation;
use crate::agents::{Agent, ManagerAgent};
use crate::envs::{EnvBuilder, EnvStructure, StatefulEnvironment};
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::thread;

/// Multi-thread simulator
pub struct MultiThreadSimulator<EB, E, MA> {
    env_builder: EB,
    // *const E to avoid indicating ownership. See:
    // https://doc.rust-lang.org/std/marker/struct.PhantomData.html#ownership-and-the-drop-check
    env_type: PhantomData<*const E>,
    manager_agent: MA,
    num_workers: usize,
}

impl<EB, E, MA> RunSimulation for MultiThreadSimulator<EB, E, MA>
where
    EB: EnvBuilder<E>,
    E: StatefulEnvironment + Send + 'static,
    MA: ManagerAgent,
    <MA as ManagerAgent>::Worker: Agent<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + 'static,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
{
    fn run_simulation(&mut self, logger: &mut dyn TimeSeriesLogger) {
        run_agent_multithread(
            &self.env_builder,
            &mut self.manager_agent,
            self.num_workers,
            logger,
        );
    }
}

fn run_worker_agent<E, WA>(environment: &mut E, worker: &mut WA)
where
    E: StatefulEnvironment + ?Sized,
    <<E as EnvStructure>::ObservationSpace as Space>::Element: Clone,
    WA: Agent<
            <<E as EnvStructure>::ObservationSpace as Space>::Element,
            <<E as EnvStructure>::ActionSpace as Space>::Element,
        > + ?Sized,
{
    // TODO: Log step statistics
    super::run_agent(environment, worker, &mut (), &mut ());
}

pub fn run_agent_multithread<EB, E, MA>(
    env_builder: &EB,
    manager: &mut MA,
    num_workers: usize,
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
{
    let mut worker_threads = vec![];
    for i in 0..num_workers {
        let env_seed = 2 * u64::try_from(i).unwrap();
        let mut env: E = env_builder.build_env(env_seed).unwrap();
        let mut worker = manager.make_worker(env_seed + 1);
        worker_threads.push(thread::spawn(move || {
            run_worker_agent(&mut env, &mut worker)
        }));
    }

    manager.run(logger);
    for thread in worker_threads.into_iter() {
        thread.join().unwrap();
    }
}
