use super::super::{
    Actor, ActorMode, Agent, AgentBuilder, BuildAgentError, ManagerAgent, SetActorMode, Step,
};
use crate::logging::{self, sync::ForwardingLogger, TimeSeriesLogger};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

impl<B, T, E> AgentBuilder<MutexAgentManager<T>, E> for B
where
    B: AgentBuilder<T, E>,
    E: ?Sized,
{
    fn build_agent(&self, env: &E, seed: u64) -> Result<MutexAgentManager<T>, BuildAgentError> {
        Ok(MutexAgentManager::new(self.build_agent(env, seed)?))
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
    receiver: Receiver<logging::sync::Message>,
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
    use crate::agents::{AgentBuilder, TabularQLearningAgent, TabularQLearningAgentConfig};
    use crate::envs::{
        DeterministicBandit, EnvBuilder, EnvWithState, FixedMeansBanditConfig, IntoStateful,
    };
    use crate::simulation;
    use crate::simulation::hooks::StepLimit;

    #[test]
    fn mutex_multithread_learns() {
        let env_config = FixedMeansBanditConfig {
            means: vec![0.0, 1.0],
        };
        let env: DeterministicBandit = env_config.build_env(0).unwrap();
        let agent_config = TabularQLearningAgentConfig::default();
        let mut agent: MutexAgentManager<TabularQLearningAgent<_, _>> =
            agent_config.build_agent(&env, 0).unwrap();
        let mut logger = ();
        let hook = StepLimit::new(1000);
        let num_workers = 5;

        simulation::run_agent_multithread::<_, EnvWithState<DeterministicBandit>, _, _>(
            &env_config,
            &mut agent,
            num_workers,
            &hook,
            &mut logger,
        );

        let agent = agent.try_into_inner().unwrap();
        let mut env = env.into_stateful(0);
        testing::eval_deterministic_bandit(agent, &mut env, 0.9);
    }
}
