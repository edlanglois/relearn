use super::super::buffers::{BuildHistoryBuffer, SerialBuffer, SerialBufferConfig};
use super::super::{
    Actor, BatchUpdate, BuildBatchUpdateActor, FullStep, SyncParams, SynchronousAgent,
};
use super::{
    BuildAgentError, BuildMultithreadAgent, InitializeMultithreadAgent, MultithreadAgentManager,
    TryIntoActor,
};
use crate::envs::{EnvStructure, StoredEnvStructure};
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;
use crossbeam_channel::{self, Receiver, Sender};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::{Arc, Mutex};

/// Multithread agent based on an inner [`BatchUpdate`] actor.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MultithreadBatchAgentConfig<TC> {
    pub actor_config: TC,
    pub buffer_config: SerialBufferConfig,
}

impl<TC, OS, AS> BuildMultithreadAgent<OS, AS> for MultithreadBatchAgentConfig<TC>
where
    TC: BuildBatchUpdateActor<OS, AS> + Clone,
    TC::BatchUpdateActor: SyncParams + Send + 'static,
    OS: Space + Clone,
    OS::Element: 'static,
    AS: Space + Clone,
    AS::Element: 'static,
    SerialBuffer<OS::Element, AS::Element>: Send,
{
    type MultithreadAgent = MultithreadBatchAgentInitializer<TC, OS, AS>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError> {
        let reference_actor = self.actor_config.build_batch_update_actor(env, seed)?;
        let (worker_send_buffer, manager_recv_buffer) = crossbeam_channel::bounded(0);
        let (manager_send_buffer, worker_recv_buffer) = crossbeam_channel::bounded(0);
        Ok(MultithreadBatchAgentInitializer {
            actor_config: self.actor_config.clone(),
            buffer_config: self.buffer_config,
            env_structure: env.into(),
            reference_actor: Arc::new(Mutex::new(reference_actor)),
            worker_seed_rng: StdRng::seed_from_u64(seed),
            num_workers: 0,
            worker_send_buffer,
            manager_recv_buffer,
            manager_send_buffer,
            worker_recv_buffer,
        })
    }
}

/// Initializer for [`MultithreadBatchAgentConfig`].
///
/// # Synchronization Model
/// The shared objects are:
/// * `RWLock<Model>` - Manager updates the model and
///                     workers synchronize their local model to the reference model.
/// * `buffer`        - Each worker has their own buffer.
///                     When they have filled it, it is sent to the manager.
///                     The manager reads and empties the buffer then sends it back.
///
/// ## Diagram
/// ```text
///                Worker              Manager
///              ==========           =========
///          {    [buffer]
/// simulate {       |
///  & fill  {       |             WriteLock(model)
///  buffer  {       *-------------|-> [buffer]      } update model
///                                |       |         } & empty buffer
///                                |       |         }
///              ReadLock(model)           |
/// sync w/  {   |                         |
/// local    {   |                         |
/// model    {   |                         |
///                                        |
///                [buffer] <--------------*
///
///  ```
///
///  The following principles collectively prevent deadlocks:
///  * A thread never attempts to acquire a lock on the model while in possession of the buffer;
///    the lock must always be acquired first (workers don't need to hold it).
///  * A thread never attempts to send the buffer while in possession of a lock on the model
///    (since the receiving thread will need to acquire the lock).
///  * A blocking [`Sender`] is used to send the buffer so the act of transferring the buffer is
///    a barrier that synchronizes both threads. This prevents a deadlock in which one thread
///    re-acquires a lock on the model before the other thread has had a chance to use it.
///
/// ## Possible Alternative
/// Instead of sending the buffer back and forth through channels, use a Mutex on the buffer with
/// a barrier for ensuring the manager and workers alternate access. One challenge with this
/// approach is that [`Barrier`](std::sync::Barrier) must be given the number of threads on
/// creation so either the number of anticipated agents would need to be given as an argument to
/// [`BuildMultithreadAgent::build_multithread_agent`] and then checked before use, or the barrier
/// would need to be sent to the workers after they are created.
#[derive(Debug)]
pub struct MultithreadBatchAgentInitializer<TC, OS, AS>
where
    TC: BuildBatchUpdateActor<OS, AS>,
    OS: Space,
    AS: Space,
{
    actor_config: TC,
    buffer_config: SerialBufferConfig,

    env_structure: StoredEnvStructure<OS, AS>,
    reference_actor: Arc<Mutex<TC::BatchUpdateActor>>, // TODO: Change back to RwLock
    worker_seed_rng: StdRng,
    num_workers: usize,

    // Separate channels for each direction so that the manager can tell when the workers have
    // exited
    worker_send_buffer: Sender<SerialBuffer<OS::Element, AS::Element>>,
    manager_recv_buffer: Receiver<SerialBuffer<OS::Element, AS::Element>>,

    manager_send_buffer: Sender<SerialBuffer<OS::Element, AS::Element>>,
    worker_recv_buffer: Receiver<SerialBuffer<OS::Element, AS::Element>>,
}

impl<TC, OS, AS> InitializeMultithreadAgent<OS::Element, AS::Element>
    for MultithreadBatchAgentInitializer<TC, OS, AS>
where
    TC: BuildBatchUpdateActor<OS, AS>,
    TC::BatchUpdateActor: SyncParams + Send + 'static,
    OS: Space + Clone,
    OS::Element: 'static,
    AS: Space + Clone,
    AS::Element: 'static,
    SerialBuffer<OS::Element, AS::Element>: Send,
{
    type Manager = MultithreadBatchManager<TC::BatchUpdateActor, OS::Element, AS::Element>;
    type Worker = MultithreadBatchWorker<TC::BatchUpdateActor, OS::Element, AS::Element>;

    fn new_worker(&mut self) -> Result<Self::Worker, BuildAgentError> {
        let mut actor = self
            .actor_config
            .build_batch_update_actor(&self.env_structure, self.worker_seed_rng.gen())?;
        // TODO: Maybe do this in the worker threads (`act`) if the synchronization is slow.
        // Would have to maintain an extra bit to indicate whether the initial sync has occurred.
        // Would also have to be careful to avoid a deadlock if the manager immediately acquires
        // a write lock on the reference actor.
        actor
            .sync_params(&self.reference_actor.lock().unwrap())
            .expect("Failed to synchronize model parameters");
        self.num_workers += 1;
        Ok(MultithreadBatchWorker {
            actor,
            reference_actor: Arc::clone(&self.reference_actor),
            buffer: Some(self.buffer_config.build_history_buffer()),
            send_buffer: self.worker_send_buffer.clone(),
            recv_buffer: self.worker_recv_buffer.clone(),
        })
    }

    fn into_manager(self) -> Self::Manager {
        MultithreadBatchManager {
            actor: self.reference_actor,
            num_workers: self.num_workers,
            send_buffer: self.manager_send_buffer,
            recv_buffer: self.manager_recv_buffer,
        }
    }

    fn boxed_into_manager(self: Box<Self>) -> Self::Manager {
        (*self).into_manager()
    }
}

#[derive(Debug)]
pub struct MultithreadBatchWorker<T, O, A> {
    actor: T,
    reference_actor: Arc<Mutex<T>>, // TODO: Change back to RwLock
    buffer: Option<SerialBuffer<O, A>>,

    send_buffer: Sender<SerialBuffer<O, A>>,
    recv_buffer: Receiver<SerialBuffer<O, A>>,
}

impl<T, O, A> Actor<O, A> for MultithreadBatchWorker<T, O, A>
where
    T: Actor<O, A>,
{
    fn act(&mut self, observation: &O) -> A {
        self.actor.act(observation)
    }

    fn reset(&mut self) {
        self.actor.reset()
    }
}

impl<T, O, A> SynchronousAgent<O, A> for MultithreadBatchWorker<T, O, A>
where
    T: Actor<O, A> + BatchUpdate<O, A> + SyncParams,
{
    fn update(&mut self, step: FullStep<O, A>, _logger: &mut dyn TimeSeriesLogger) {
        let full = self.buffer.as_mut().unwrap().push(step);
        if full {
            // Send buffer to manager to update the reference actor
            self.send_buffer.send(self.buffer.take().unwrap()).unwrap();

            // Acquiring the read lock ensures that the manager has finished its  updates
            // because the manager will acquire a write lock on the reference actor before
            // accepting buffers from the workers.
            self.actor
                .sync_params(&self.reference_actor.lock().unwrap())
                .expect("Failed to synchronize model parameters");

            // Get a buffer back from the manger
            self.buffer = Some(self.recv_buffer.recv().unwrap());
        }
    }
}

#[derive(Debug)]
pub struct MultithreadBatchManager<T, O, A> {
    actor: Arc<Mutex<T>>,
    num_workers: usize,

    send_buffer: Sender<SerialBuffer<O, A>>,
    recv_buffer: Receiver<SerialBuffer<O, A>>,
}

impl<T, O, A> MultithreadAgentManager for MultithreadBatchManager<T, O, A>
where
    T: BatchUpdate<O, A>,
{
    fn run(&mut self, logger: &mut dyn TimeSeriesLogger) {
        loop {
            // Acquire a write lock on the reference actor
            let mut actor = self.actor.lock().unwrap();

            // Get full buffers from the worker threads.
            let mut buffers: Vec<_> = self.recv_buffer.iter().take(self.num_workers).collect();

            if buffers.is_empty() {
                // Worker threads have exited
                return;
            }
            assert_eq!(buffers.len(), self.num_workers);

            actor.batch_update(&mut buffers, logger);
            drop(actor);

            // Clear buffers and return to the agents.
            for mut buffer in buffers.into_iter() {
                buffer.clear();
                self.send_buffer.send(buffer).unwrap();
            }
        }
    }
}

impl<T, O, A> TryIntoActor for MultithreadBatchManager<T, O, A> {
    type Actor = T;

    fn try_into_actor(self) -> Result<Self::Actor, Self> {
        match Arc::try_unwrap(self.actor) {
            Ok(rwlock) => Ok(rwlock.into_inner().unwrap()),
            Err(actor) => Err(Self { actor, ..self }),
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
    fn multithread_batch_learns() {
        let env_config = DeterministicBandit::from_means(vec![0.0, 1.0]).unwrap();
        let agent_config = MultithreadBatchAgentConfig {
            actor_config: TabularQLearningAgentConfig::default(),
            buffer_config: SerialBufferConfig {
                soft_threshold: 10,
                hard_threshold: 11,
            },
        };
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
