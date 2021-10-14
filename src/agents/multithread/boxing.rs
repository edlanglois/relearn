//! Multithread initializer wrapper that builds boxed trait objects.
use super::super::Agent;
use super::{InitializeMultithreadAgent, MultithreadAgentManager};

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
    type Worker = Box<dyn Agent<O, A> + Send>;

    fn make_worker(&self, id: usize) -> Self::Worker {
        Box::new(self.inner.make_worker(id))
    }

    fn into_manager(self) -> Self::Manager {
        Box::new(self.inner.into_manager())
    }
}
