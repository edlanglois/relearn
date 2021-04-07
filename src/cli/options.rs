//! Command-line options
use super::agent::AgentName;
use super::env::EnvName;
use super::optimizer::OptimizerName;
use super::policy::PolicyName;
use crate::torch::Activation;
use clap::{crate_authors, crate_description, crate_version, Clap};

#[derive(Clap, Debug)]
#[clap(
    version = crate_version!(),
    author = crate_authors!(),
    about = crate_description!(),
)]
pub struct Options {
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

    #[clap(long)]
    /// Agent exploration rate
    pub exploration_rate: Option<f64>,

    #[clap(long)]
    /// Number of steps the agent collects between policy updates.
    pub steps_per_epoch: Option<usize>,

    #[clap(long)]
    /// Number of samples for Thompson sampling agents.
    pub num_samples: Option<usize>,

    // Policy args
    #[clap(long, arg_enum, default_value = "mlp")]
    /// Policy name
    // Note: If using Option<PolicyName> instead, PolicyDef::default() can't be used because
    // we need to read the inner attributes from Options if they are set.
    pub policy: PolicyName,

    /// Policy mlp activation function
    #[clap(long, arg_enum)]
    pub activation: Option<Activation>,

    #[clap(long)]
    /// Policy mlp hidden layer sizes
    pub hidden_sizes: Option<Vec<usize>>,

    #[clap(long)]
    /// Policy rnn hidden layer size
    pub rnn_hidden_size: Option<usize>,

    #[clap(long)]
    /// Policy rnn number of hidden layers
    pub rnn_num_layers: Option<usize>,

    #[clap(long, arg_enum)]
    /// Policy rnn output activation function
    pub rnn_output_activation: Option<Activation>,

    // Optimizer args
    #[clap(long, arg_enum, default_value = "adam")]
    /// Optimizer name
    pub optimizer: OptimizerName,

    #[clap(long)]
    /// Agent optimizer learning rate
    pub learning_rate: Option<f64>,

    #[clap(long)]
    /// Agent optimizer momentum
    pub momentum: Option<f64>,

    #[clap(long)]
    /// Agent optimizer weight decay (L2 regularization)
    pub weight_decay: Option<f64>,

    // Simulation args
    #[clap(long)]
    /// Maximum number of experiment steps
    pub max_steps: Option<u64>,
}
