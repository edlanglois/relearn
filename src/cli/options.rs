//! Command-line options
use super::agent::AgentType;
use super::env::{BanditArmPrior, EnvType};
use super::optimizer::OptimizerType;
use super::policy::PolicyType;
use crate::torch::Activation;
use clap::{crate_authors, crate_description, crate_version, Clap};

#[derive(Clap, Debug)]
#[clap(
    version = crate_version!(),
    author = crate_authors!(),
    about = crate_description!(),
    after_help = "Most options only apply for some environments/agents and are ignored otherwise.",
)]
pub struct Options {
    // Environment options
    #[clap(arg_enum)]
    /// Environment type
    pub environment: EnvType,

    #[clap(long, help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// Number of states in the environment
    pub num_states: Option<u64>,

    #[clap(long, help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// Number of actions in the environment
    pub num_actions: Option<u64>,

    #[clap(long, help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// Length of episodes in the environment
    pub episode_len: Option<u64>,

    #[clap(long, help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// Environment discount factor
    pub discount_factor: Option<f64>,

    #[clap(long, help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// Bandit arm reward means
    pub arm_rewards: Option<Vec<f64>>,

    #[clap(long, arg_enum, default_value = "fixed", help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// How bandit arm reward means are generated (once). Fixed uses --arm-rewards.
    pub arm_prior: BanditArmPrior,

    // Agent args
    #[clap(arg_enum)]
    /// Agent type
    pub agent: AgentType,

    #[clap(long, help_heading = Some("AGENT OPTIONS"))]
    /// Agent exploration rate
    pub exploration_rate: Option<f64>,

    #[clap(long, help_heading = Some("AGENT OPTIONS"))]
    /// Number of steps the agent collects between policy updates.
    pub steps_per_epoch: Option<usize>,

    #[clap(long, help_heading = Some("AGENT OPTIONS"))]
    /// Number of samples for Thompson sampling agents.
    pub num_samples: Option<usize>,

    // Policy options
    #[clap(long, arg_enum, default_value = "mlp", help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy type
    // Note: If using Option<PolicyType> instead, PolicyDef::default() can't be used because
    // we need to read the inner attributes from Options if they are set.
    pub policy: PolicyType,

    /// Policy MLP activation function
    #[clap(long, arg_enum, help_heading = Some("AGENT POLICY OPTIONS"))]
    pub activation: Option<Activation>,

    #[clap(long, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy MLP hidden layer sizes
    pub hidden_sizes: Option<Vec<usize>>,

    #[clap(long, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy rnn hidden layer size
    pub rnn_hidden_size: Option<usize>,

    #[clap(long, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy rnn number of hidden layers
    pub rnn_num_layers: Option<usize>,

    #[clap(long, arg_enum, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy rnn output activation function
    pub rnn_output_activation: Option<Activation>,

    // Optimizer options
    #[clap(long, arg_enum, default_value = "adam", help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Optimizer type
    pub optimizer: OptimizerType,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Agent optimizer learning rate
    pub learning_rate: Option<f64>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Agent optimizer momentum
    pub momentum: Option<f64>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Agent optimizer weight decay (L2 regularization)
    pub weight_decay: Option<f64>,

    // Simulation options
    #[clap(long, default_value = "1", help_heading = Some("SIMULATION OPTIONS"))]
    /// Random seed for the experiment
    pub seed: u64,

    #[clap(long, help_heading = Some("SIMULATION OPTIONS"))]
    /// Maximum number of experiment steps
    pub max_steps: Option<u64>,
}
