/// Builds loggers for new threads.
pub trait BuildThreadLogger {
    type ThreadLogger: Send + 'static;

    /// Build a logger for a new thread
    ///
    /// # Args
    /// * `thread_id` - Unique identifier for the thread.
    fn build_thread_logger(&self, thread_id: usize) -> Self::ThreadLogger;
}
