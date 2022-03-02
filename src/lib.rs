//! A reinforcement learning library.
//!
//! This library defines a set of [environments](crate::envs) and [learning agents](crate::agents)
//! and [simulates](crate::simulation) their interaction.
//!
//! Environments implement the [`Environment`](crate::envs::Environment) trait, which has
//! associated observation, action, and state types.
//! Agents implement [`Agent`] and provide [`Actors`](crate::agents::Actor) that generate actions
//! in response to environment observations.
//! Agents can learn via the [`BatchUpdate`](crate::agents::BatchUpdate`) trait.
//!
//! Agent traits are generic over the observation (`O`) and action (`A`) types of the environment.
//! The [`EnvStructure`] trait provides more details about possible values for these types via the
//! [`Space`](crate::spaces::Space) trait. A `Space` can be thought of as a runtime-defined type,
//! describing a set of possible values while methods are provided by other traits in
//! [`spaces`](crate::spaces).
//!
//! Basic actor-environment simulation (without training) is performed by
//! [`SimulatorSteps`](crate::simulation::SimulatorSteps).
//! This is an iterator over partial [`Step`](crate::simulation::Step) and allows more
//! complex iteration via
//! [`SimulatorSteps::step_with`](crate::simulation::SimulatorSteps::step_with).
//! Training is performed by [`train_serial`](crate::simulation::train_serial) and
//! [`train_parallel`](crate::simulation::train_parallel).
//!
//! This library uses [PyTorch](https://pytorch.org/) via [tch].
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::doc_markdown)]
#![warn(clippy::explicit_iter_loop)]
#![warn(clippy::for_kv_map)] // part of warn(clippy::all), specifically style?
#![warn(clippy::missing_const_for_fn)] // has some false positives
#![warn(clippy::needless_borrow)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::redundant_closure_for_method_calls)]
#![warn(clippy::use_self)] // also triggered by macro expansions

/// Allow referring to this crate as `relearn` so that `relearn_derive` macros  will work both in
/// this crate and by third-party crates that use `relearn`.
/// See <https://stackoverflow.com/a/57049687/1267562>.
extern crate self as relearn;

pub mod agents;
pub mod envs;
pub mod logging;
pub mod simulation;
pub mod spaces;
pub mod torch;
pub mod utils;

pub use agents::{Actor, Agent, BatchUpdate, BuildAgent};
pub use envs::{BuildEnv, EnvStructure, Environment};
pub use simulation::{train_parallel, train_serial, Step};
pub use utils::save::SaveLoad;

/// Pseudo-random number generator type used by agents and environments in this crate.
///
/// This is a cryptographically secure PRNG to ensure that [`rand::SeedableRng::from_rng`]
/// can be used to fork random generators without concern for correlations.
/// Cryptographic security is not otherwise important so the number of rounds used is on the low
/// end.
pub type Prng = rand_chacha::ChaCha8Rng;

#[allow(unused_imports)]
#[macro_use]
extern crate relearn_derive;
