//! Torch components
#[macro_use]
mod macros;
pub mod agents;
pub mod backends;
pub mod critic;
pub mod distributions;
pub mod history;
pub mod modules;
pub mod optimizers;
pub mod seq_modules;
pub mod updaters;
pub mod utils;

pub use modules::{Activation, ModuleBuilder};
pub use optimizers::{Optimizer, OptimizerBuilder};
pub use seq_modules::{Gru, Lstm};
