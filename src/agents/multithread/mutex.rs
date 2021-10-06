use super::super::{
    Actor, ActorMode, Agent, BuildAgent, BuildAgentError, BuildManagerAgent, ManagerAgent,
    SetActorMode, Step,
};
use crate::envs::EnvStructure;
use crate::logging::{self, ForwardingLogger, TimeSeriesLogger};
use crate::spaces::Space;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

impl<AC, OS, AS> BuildManagerAgent<OS, AS> for MutexAgentConfig<AC>
where
    AC: BuildAgent<OS, AS>,
    AC::Agent: Send + 'static,
    OS: Space,
    AS: Space,
{
    type ManagerAgent = MutexAgentManager<AC::Agent>;

    fn build_manager_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::ManagerAgent, BuildAgentError> {
        Ok(MutexAgentManager::new(
            self.agent_config.build_agent(env, seed)?,
        ))
    }
}

/// A mutex-based "multithread" manager agent.
///
/// This wraps any agent to act like a multithread agent by sharing a single copy of the agent with
/// a mutex. This is a very inefficient way to make a multithread agent but serves as a simple
/// way to test multi-thread simulation.
///
/// Logging works by forwarding messages to the manager thread where they are logged by the logger
/// passed to `ManagerAgent::run`.
#[derive(Debug)]
pub struct MutexAgentManager<T> {
    /// Inner agent
    agent: Arc<Mutex<T>>,
    /// Forwarding logger to use in the workers.
    logger: ForwardingLogger,
    /// Receives log messages from the workers.
    receiver: Receiver<logging::forwarding::Message>,
}

/// A mutex-based "multithread" agent worker.
///
/// This wraps any agent to act like a multithread worker by sharing a single copy of the agent
/// with a mutex. Created by [`MutexAgentManager`].
#[derive(Debug)]
pub struct MutexAgentWorker<T> {
    /// Inner agent
    agent: Arc<Mutex<T>>,
    /// Logger that forwards messages to the manager's receiver.
    logger: ForwardingLogger,
}

impl<T> MutexAgentManager<T> {
    pub fn new(agent: T) -> Self {
        let (logger, receiver) = ForwardingLogger::new();
        Self {
            agent: Arc::new(Mutex::new(agent)),
            logger,
            receiver,
        }
    }

    /// Try to extract the inner agent.
    ///
    /// Fails if any workers exist. Returns the manager back on failure.
    pub fn try_into_inner(self) -> Result<T, Self> {
        match Arc::try_unwrap(self.agent) {
            Ok(mutex) => Ok(mutex.into_inner().unwrap()),
            Err(agent) => Err(Self {
                agent,
                logger: self.logger,
                receiver: self.receiver,
            }),
        }
    }
}

impl<T: Actor<O, A>, O, A> Actor<O, A> for MutexAgentWorker<T> {
    fn act(&mut self, observation: &O, new_episode: bool) -> A {
        self.agent.lock().unwrap().act(observation, new_episode)
    }
}

impl<T: Agent<O, A>, O, A> Agent<O, A> for MutexAgentWorker<T> {
    fn update(&mut self, step: Step<O, A>, _: &mut dyn TimeSeriesLogger) {
        self.agent.lock().unwrap().update(step, &mut self.logger)
    }
}

impl<T: SetActorMode> SetActorMode for MutexAgentManager<T> {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.agent.lock().unwrap().set_actor_mode(mode)
    }
}

impl<T: SetActorMode> SetActorMode for MutexAgentWorker<T> {
    fn set_actor_mode(&mut self, mode: ActorMode) {
        self.agent.lock().unwrap().set_actor_mode(mode)
    }
}

impl<T: Send + 'static> ManagerAgent for MutexAgentManager<T> {
    type Worker = MutexAgentWorker<T>;

    fn make_worker(&mut self, _seed: u64) -> Self::Worker {
        MutexAgentWorker {
            agent: Arc::clone(&self.agent),
            logger: self.logger.clone(),
        }
    }

    fn run(&mut self, logger: &mut dyn TimeSeriesLogger) {
        // Run while any workers exist
        loop {
            match self.receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(message) => message.log(logger).unwrap(),
                Err(RecvTimeoutError::Timeout) => {
                    if Arc::strong_count(&self.agent) <= 1 {
                        // Only the copy held by the manager is left => all workers done
                        return;
                    }
                }
                Err(_) => unreachable!(), // This thread holds a copy of send, channel never closes
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::testing;
    use super::*;
    use crate::agents::{BuildManagerAgent, TabularQLearningAgentConfig};
    use crate::envs::{DeterministicBandit, IntoEnv};
    use crate::simulation;
    use crate::simulation::hooks::StepLimitConfig;
    use std::sync::{Arc, RwLock};

    #[test]
    fn mutex_multithread_learns() {
        let env = DeterministicBandit::from_means(vec![0.0, 1.0]).unwrap();
        let agent_config = MutexAgentConfig::new(TabularQLearningAgentConfig::default());
        let mut agent = agent_config.build_manager_agent(&env, 0).unwrap();
        let mut logger = ();
        let worker_hook_config = StepLimitConfig::new(1000);
        let num_workers = 5;

        let locked_env = Arc::new(RwLock::new(env));
        simulation::run_agent_multithread(
            &locked_env,
            &mut agent,
            num_workers,
            &worker_hook_config,
            0,
            0,
            &mut logger,
        );

        let agent = agent.try_into_inner().unwrap();
        let env = Arc::try_unwrap(locked_env).unwrap().into_inner().unwrap();
        let mut env = env.into_env(0);
        testing::eval_deterministic_bandit(agent, &mut env, 0.9);
    }
}
