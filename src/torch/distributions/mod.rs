//! Torch statistical distributions
//!
//! These types are all capable of representing a batch of distributions, with arbitrary batch
//! shape. The return values of statistics methods are tensors with the same batch shape.
mod categorical;
mod deterministic;

pub use categorical::Categorical;
pub use deterministic::DeterministicEmptyVec;
