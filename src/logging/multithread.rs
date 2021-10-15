/// Builds loggers for new threads.
pub trait BuildThreadLogger {
    type ThreadLogger;

    /// Build a logger for a new thread
    ///
    /// # Args
    /// * `thread_id` - Unique identifier for the thread.
    fn build_thread_logger(&self, thread_id: usize) -> Self::ThreadLogger;
}

impl BuildThreadLogger for () {
    type ThreadLogger = ();

    fn build_thread_logger(&self, _thread_id: usize) -> Self::ThreadLogger {}
}
