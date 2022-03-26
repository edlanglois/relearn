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
//! Environment-actor simulation is performed by [`Steps`](crate::simulation::Steps) and the
//! resulting [`Step`](crate::simulation::Step) are accessible via an `Iterator` interface.
//! Training is performed by [`train_serial`](crate::simulation::train_serial) and
//! [`train_parallel`](crate::simulation::train_parallel).
//!
//! This library uses [PyTorch](https://pytorch.org/) via [tch].
#![warn(clippy::pedantic)]
// Nursery level
#![warn(clippy::missing_const_for_fn)] // has some false positives
#![warn(clippy::use_self)] // also triggered by macro expansions
// Excluded Pedantic Lints
#![allow(
    clippy::cast_precision_loss,  // The precision loss is often expected
    clippy::default_trait_access, // Alternative can be complex types, not more clear
    clippy::enum_glob_use,        // Use Enum globs in match statements
    clippy::let_underscore_drop,  // Typical of Tensor in-place ops.
    clippy::missing_errors_doc,   // Errors obvious or complex --- easier to look at error type
    clippy::missing_panics_doc,   // Maybe be logically impossible or only in extreme cases
    clippy::module_name_repetitions, // Types pub exported in different modeule.
    clippy::semicolon_if_nothing_returned, // Conceptually returning "result" of inner operation
    clippy::similar_names,        // Sometimes prefer names that are similar. Consider removing.
    clippy::single_match_else,    // Match statement can be more readable.
    clippy::type_repetition_in_bounds, // Frequent false positives
    clippy::wildcard_imports,     // Used in test modules
)]

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
pub use simulation::{train_parallel, train_serial, Simulation, Step, Steps, StepsIter};

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
