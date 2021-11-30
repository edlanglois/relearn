//! Multi-thread agents
mod batch;
mod mutex;

pub use batch::{
    MultithreadBatchAgentConfig, MultithreadBatchAgentInitializer, MultithreadBatchManager,
    MultithreadBatchWorker,
};
pub use mutex::{MutexAgentConfig, MutexAgentInitializer, MutexAgentManager, MutexAgentWorker};

use super::{BuildAgentError, SynchronousAgent};
use crate::envs::EnvStructure;
use crate::logging::TimeSeriesLogger;
use crate::spaces::Space;

/// Build a multithread agent initializer ([`InitializeMultithreadAgent`]).
pub trait BuildMultithreadAgent<OS: Space, AS: Space> {
    type MultithreadAgent: InitializeMultithreadAgent<OS::Element, AS::Element>;

    fn build_multithread_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        seed: u64,
    ) -> Result<Self::MultithreadAgent, BuildAgentError>;
}

/// Initialize a multithread agent.
///
/// A multithread agent consists of a single manager and multiple workers.
/// The manager is run on the current thread while each worker is sent to run on its own thread.
/// The manager and workers are responsible for internaly coordinating updates and synchronization.
pub trait InitializeMultithreadAgent<O, A> {
    type Manager: MultithreadAgentManager;
    type Worker: SynchronousAgent<O, A> + Send + 'static;

    /// Create a new worker instance.
    fn new_worker(&mut self) -> Result<Self::Worker, BuildAgentError>;

    /// Convert the initializer into the manager instance.
    fn into_manager(self) -> Self::Manager;

    /// Convert a boxed version of self into the manager instance
    ///
    /// This is to [help with implementing][1] [`InitializeMultithreadAgent`] for
    /// `Box<dyn InitializeMultithreadAgent>` without [unsized r-values][2].
    ///
    /// [1]: https://stackoverflow.com/a/46625787/1267562
    /// [2]: https://rust-lang.github.io/rfcs/1909-unsized-rvalues.html
    ///
    /// As far as I can tell, it is not possible to provide a default implementation without
    /// requiring `Self: Sized`, which prevents it from being used on usized Self as intended.
    /// Implementors can define it as `(*self).into_manager()`.
    fn boxed_into_manager(self: Box<Self>) -> Self::Manager;
}

impl<T, O, A> InitializeMultithreadAgent<O, A> for Box<T>
where
    T: InitializeMultithreadAgent<O, A> + ?Sized,
{
    type Manager = T::Manager;
    type Worker = T::Worker;
    fn new_worker(&mut self) -> Result<Self::Worker, BuildAgentError> {
        T::new_worker(self)
    }
    fn into_manager(self) -> Self::Manager {
        self.boxed_into_manager()
    }
    fn boxed_into_manager(self: Box<Self>) -> Self::Manager {
        // Note: This is self: Box<Box<T>> so probably it shouldn't ever be called.
        (*self).into_manager()
    }
}

/// The manager part of a multithread agent.
pub trait MultithreadAgentManager {
    fn run(&mut self, logger: &mut dyn TimeSeriesLogger);
}

impl<T> MultithreadAgentManager for Box<T>
where
    T: MultithreadAgentManager + ?Sized,
{
    fn run(&mut self, logger: &mut dyn TimeSeriesLogger) {
        T::run(self, logger)
    }
}

/// Try to convert into a stand-alone actor.
///
/// This is generally implemented for [`MultithreadAgentManager`].
pub trait TryIntoActor: Sized {
    type Actor;

    /// Try to convert the manager into a standalone actor, otherwise return self
    ///
    /// For multithread managers, this is likely to fail if any workers still exist.
    fn try_into_actor(self) -> Result<Self::Actor, Self>;
}

/// Multithread initializer that boxes the manager and workers.
pub struct BoxingMultithreadInitializer<T> {
    inner: T,
}
impl<T> BoxingMultithreadInitializer<T> {
    pub const fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T, O, A> InitializeMultithreadAgent<O, A> for BoxingMultithreadInitializer<T>
where
    T: InitializeMultithreadAgent<O, A>,
    T::Manager: 'static,
    O: 'static,
    A: 'static,
{
    type Manager = Box<dyn MultithreadAgentManager>;
    type Worker = Box<dyn SynchronousAgent<O, A> + Send>;

    fn new_worker(&mut self) -> Result<Self::Worker, BuildAgentError> {
        Ok(Box::new(self.inner.new_worker()?))
    }
    fn into_manager(self) -> Self::Manager {
        Box::new(self.inner.into_manager())
    }
    fn boxed_into_manager(self: Box<Self>) -> Self::Manager {
        (*self).into_manager()
    }
}
