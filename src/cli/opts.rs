//! Command-line options
use super::agent::AgentName;
use super::env::EnvName;
use clap::{crate_authors, crate_description, crate_version, Clap};

#[derive(Clap, Debug)]
#[clap(
    version = crate_version!(),
    author = crate_authors!(),
    about = crate_description!(),
)]
pub struct Opts {
    #[clap(long, default_value = "1")]
    /// Random seed for the experiment
    pub seed: u64,
    // Environment args
    #[clap(arg_enum)]
    /// Environment name
    pub environment: EnvName,

    #[clap(long)]
    /// Number of states in the environment; when configurable
    pub num_states: Option<u32>,

    #[clap(long)]
    /// Number of actions in the environment; when configurable
    pub num_actions: Option<u32>,

    #[clap(long)]
    /// Environment discount factor; when configurable
    pub discount_factor: Option<f64>,

    // Agent args
    #[clap(arg_enum)]
    /// Agent name
    pub agent: AgentName,

    #[clap(long, default_value = "0.01")]
    /// Agent learning rate
    pub learning_rate: f64,

    #[clap(long, default_value = "0.2")]
    /// Agent exploration rate
    pub exploration_rate: f64,

    #[clap(long, default_value = "1000")]
    /// Number of steps the agent collects between policy updates.
    pub steps_per_epoch: usize,

    #[clap(long, default_value = "1")]
    pub num_samples: usize,

    // Experiment args
    #[clap(long)]
    /// Maximum number of experiment steps
    pub max_steps: Option<u64>,
}
