//! Command-line interface
mod agent;
mod env;
mod optimizer;
mod options;
mod policy;

pub use options::Options;

/// Can be updated from a value of type T.
pub trait Update<T> {
    /// Update in-place from the given source value.
    fn update(&mut self, source: T);
}

/// Can be updated from a value of type T.
pub trait WithUpdate<T> {
    /// Apply an update from the given source value.
    fn with_update(self, source: T) -> Self;
}

impl<T, U: Update<T>> WithUpdate<T> for U {
    fn with_update(mut self, source: T) -> Self {
        self.update(source);
        self
    }
}
