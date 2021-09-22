//! A reinforcement learning library.
//!
//! # Overview
//! This library defines a set of [environments](crate::envs) and [learning agents](crate::agents)
//! and [simulates](crate::simulation) their interaction.
//!
//! It uses [PyTorch](https://pytorch.org/) via [tch].
//!
//! ## Environments
//! A reinforcement learning [`Environment`] is an environment structure with internal state.
//! The fundamental operation is to take a [step](Environment::step) from a state given
//! some action, resulting in a successor state, a reward value, and a flag indicating whether the
//! current _episode_ is done. The [`Step`](crate::agents::Step) structure stores a
//! description of the observable parts of an environment step.
//!
//! ### Episode
//! A sequence of environment steps each starting from the successor state of the previous.
//! The initial state is set by calling [`Environment::reset`].
//! An episode ends when [`Environment::step`] sets the `episode_done` flag in its return
//! value.
//! An episode may end on a _terminal state_ in which case all future rewards are assumed to be
//! zero. If instead the final state is non-terminal then there may have been non-zero future
//! rewards if the episode had continued.
//!
//! ### Terminal State
//! An environment state that immediately ends the _episode_ with 0 future reward.
//! From the perspective of the MDP formalism (in which all episodes are infinitely long),
//! a state from which all steps, no matter the action,
//! have 0 reward and lead to another terminal state.
//!
//! ### Return
//! The discounted sum of future rewards (`return = sum_i { reward_i * discount_factor ** i }`).
//! May refer to the rewards of an entire episode or the future rewards from a particular step.
//!
//! ### Space
//! A space is a mathematical set with some added structure,
//! used here for defining the set of possible actions and observations
//! of a reinforcement learning environment.
//!
//! The core interface is the [`Space`] trait
//! with additional functionality provided by other traits in [`spaces`](crate::spaces).
//! The actual elements of a space have type `Space::Element`.
//!
//! [`Space`]: crate::spaces::Space
//!
//! ### Action Space
//! A set ([`EnvStructure::ActionSpace`]) containing all possible actions for the environment.
//! The action space is independent of the environment state so every action in the space is
//! allowed in any state.
//! An invalid action may be simulated by providing low reward and ending the episode.
//!
//! ### Observation Space
//! A set ([`EnvStructure::ObservationSpace`])
//! containing all possible observations an environment might produce.
//! May contain elements that cannot be produced as observations.
//! To be more precise, the set of possible observations is actually
//! `Option<ObservationSpace>` where `None` represents any _terminal states_.
//!
//! ## Agents
//! An [`Actor`] interacts with an environment.
//! An [`Agent`] is an Actor with the ability to persistently update.
//! An Actor may "learn" within an episode by conditioning on the observed episode history, but
//! only [`Agent::update`] allows learning across episodes.
//!
//! ### Policy
//! A policy maps a sequence of episode history features to parameters of an action distribution
//! for the current state. A policy may use the past within an episode but not across episodes and
//! not from the future.
//!
//! ### Critic
//! A critic assigns a value to each step in an episode. It does so retroactively with full access
//! to the episode future. It may also depend on the past history within an episode. It may not
//! depend on information between episodes. The value is not necessarily the (expected) return from
//! a given state but should be correlated with expected return such that higher values indicate
//! better states and actions.
//!
//! The critic is used for generating training targets when updating the policy. Examples include
//! the [empirical return](crate::torch::critic::Return)
//! and [Generalized Advantage Estimation](crate::torch::critic::Gae).
//!
//! This usage is possibly non-standard. I have not found it clear whether the standard use of
//! "critic" refers exclusively to value estimates using only the history or if retroactive value
//! estimates can be included.
//!
//! ### Value Function
//! A function approximator that maps a sequence of episode history features to estimates of the
//! expected future return of each observation or observation-action pair.
//! May only use the past history within an episode, not from the future or across episodes.
//! Some critics use value functions to improve their value estimates.
//!
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
pub mod agents;
pub mod cli;
pub mod defs;
pub mod envs;
mod error;
pub mod logging;
pub mod simulation;
pub mod spaces;
pub mod torch;
pub mod utils;

pub use agents::{Actor, Agent, Step};
pub use defs::{AgentDef, EnvDef, MultiThreadAgentDef, OptimizerDef, SeqModDef};
pub use envs::{EnvStructure, Environment};
pub use error::RLError;
pub use simulation::{run_actor, run_agent, RunSimulation};
