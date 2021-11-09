use super::super::{Actor, Agent, BuildAgent, BuildAgentError, Step};
use super::{
    BuildMultithreadAgent, InitializeMultithreadAgent, MultithreadAgentManager, TryIntoActor,
};
use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;
use std::sync::{Arc, Mutex};

/// Configuration for [`MutexAgentManager`] and [`MutexAgentWorker`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MutexAgentConfig<AC> {
    pub agent_config: AC,
}

impl<AC> MutexAgentConfig<AC> {
    pub const fn new(agent_config: AC) -> Self {
        Self { agent_config }
    }
}

impl<AC> From<AC> for MutexAgentConfig<AC> {
    fn from(agent_config: AC) -> Self {
        Self { agent_config }
    }
}

impl<AC, OS, AS> BuildMultithreadAgent<OS, AS> for MutexAgentConfig<AC>
where
    AC: BuildAgent<OS, AS>,
    AC::Agent: Send + 'static,
    OS: Space,
    AS: Space,
{
    type MultithreadAgent = MutexAgentInitializer<AC::Agent>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        Ok(MutexAgentInitializer::new(
            self.agent_config.build_agent(env, seed)?,
        ))
    }
}

/// A mutex-based "multithread" agent.
///
/// This wraps any agent to act like a multithread agent by sharing a single copy of the agent with
/// a mutex. This is a **very inefficient** way to make a multithread agent but serves as a simple
/// way to test multi-thread simulation.
#[derive(Debug, Default)]
pub struct MutexAgentInitializer<T> {
    /// Inner agent
    pub agent: Arc<Mutex<T>>,
}

impl<T> MutexAgentInitializer<T> {
    pub fn new(agent: T) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
        }
    }
}

impl<T, O, A> InitializeMultithreadAgent<O, A> for MutexAgentInitializer<T>
where
    T: Agent<O, A> + Send + 'static,
{
    type Manager = MutexAgentManager<T>;
    type Worker = MutexAgentWorker<T>;

    fn new_worker(&mut self) -> Result<Self::Worker, BuildAgentError> {
        Ok(MutexAgentWorker {
            agent: Arc::clone(&self.agent),
        })
    }

    fn into_manager(self) -> Self::Manager {
        MutexAgentManager {
            agent: Arc::clone(&self.agent),
        }
    }

    fn boxed_into_manager(self: Box<Self>) -> Self::Manager {
        (*self).into_manager()
    }
}

/// A mutex-based multithread agent worker.
///
/// See [`MutexAgentInitializer`].
#[derive(Debug, Default)]
pub struct MutexAgentWorker<T> {
    pub agent: Arc<Mutex<T>>,
}

impl<T, O, A> Actor<O, A> for MutexAgentWorker<T>
where
    T: Actor<O, A>,
{
    fn act(&mut self, observation: &O, new_episode: bool) -> A {
        self.agent.lock().unwrap().act(observation, new_episode)
    }
}

impl<T, O, A> Agent<O, A> for MutexAgentWorker<T>
where
    T: Agent<O, A>,
{
    fn update(&mut self, step: Step<O, A>, logger: &mut dyn TimeSeriesLogger) {
        self.agent.lock().unwrap().update(step, logger)
    }
}

/// A mutex-based multithread agent manager.
///
/// See [`MutexAgentInitializer`].
#[derive(Debug, Default)]
pub struct MutexAgentManager<T> {
    pub agent: Arc<Mutex<T>>,
}

impl<T> MultithreadAgentManager for MutexAgentManager<T> {
    fn run(&mut self, _logger: &mut dyn TimeSeriesLogger) {}
}

impl<T> TryIntoActor for MutexAgentManager<T> {
    type Actor = T;

    fn try_into_actor(self) -> Result<Self::Actor, Self> {
        match Arc::try_unwrap(self.agent) {
            Ok(mutex) => Ok(mutex.into_inner().unwrap()),
            Err(agent) => Err(Self { agent }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::testing;
    use super::*;
    use crate::agents::TabularQLearningAgentConfig;
    use crate::envs::{BuildEnv, DeterministicBandit};
    use crate::simulation::hooks::StepLimitConfig;
    use crate::simulation::MultithreadSimulatorConfig;

    #[test]
    fn mutex_multithread_learns() {
        let env_config = DeterministicBandit::from_means(vec![0.0, 1.0]).unwrap();
        let agent_config = MutexAgentConfig::new(TabularQLearningAgentConfig::default());
        let hook_config = StepLimitConfig::new(1000);
        let worker_logger_config = ();
        let sim_config = MultithreadSimulatorConfig { num_workers: 3 };

        let simulator = sim_config.build_simulator(
            env_config.clone(),
            agent_config,
            hook_config,
            worker_logger_config,
        );
        let manager = simulator.train(0, 0, &mut ()).unwrap();

        let agent = manager.try_into_actor().unwrap();
        let mut env = env_config.build_env(1).unwrap();
        testing::eval_deterministic_bandit(agent, &mut env, 0.9);
    }
}
