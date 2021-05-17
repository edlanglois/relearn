//! Command-line options
use super::agent::AgentType;
use super::env::{BanditArmPrior, EnvType};
use super::optimizer::{OptimizerOptions, OptimizerType};
use super::seq_mod::SeqModType;
use super::step_value::StepValueType;
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
    #[clap(long, arg_enum, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy type
    pub policy: Option<SeqModType>,

    /// Policy MLP activation function
    #[clap(long, arg_enum, help_heading = Some("AGENT POLICY OPTIONS"))]
    pub activation: Option<Activation>,

    #[clap(long, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy MLP hidden layer sizes
    pub hidden_sizes: Option<Vec<usize>>,

    #[clap(long, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy rnn hidden layer size
    pub rnn_hidden_size: Option<usize>,

    #[clap(long, arg_enum, help_heading = Some("AGENT POLICY OPTIONS"))]
    /// Policy rnn output activation function
    pub rnn_output_activation: Option<Activation>,

    // Value function options
    #[clap(long, arg_enum, help_heading = Some("AGENT VALUE FN OPTIONS"))]
    /// Agent step value type
    pub step_value: Option<StepValueType>,

    #[clap(long, help_heading = Some("AGENT VALUE FN OPTIONS"))]
    /// Maximum discount factor used by GAE
    pub gae_discount_factor: Option<f64>,

    #[clap(long, help_heading = Some("AGENT VALUE FN OPTIONS"))]
    /// Lambda interpolation factor used by GAE
    pub gae_lambda: Option<f64>,

    // Optimizer options
    #[clap(long, arg_enum, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Agent optimizer(s) type
    pub optimizer: Option<OptimizerType>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Agent optimizer(s) learning rate
    pub learning_rate: Option<f64>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Agent optimizer(s) momentum
    pub momentum: Option<f64>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Agent optimizer(s) weight decay (L2 regularization)
    pub weight_decay: Option<f64>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Max policy KL divergence per update step; for trust-region methods.
    pub max_policy_step_kl: Option<f64>,

    #[clap(long, arg_enum, help_heading = Some("AGENT VALUE FN OPTIMIZER OPTIONS"))]
    /// Agent value function optimizer type
    pub value_fn_optimizer: Option<OptimizerType>,

    #[clap(long, help_heading = Some("AGENT VALUE FN OPTIMIZER OPTIONS"))]
    /// Agent value function optimizer learning rate
    pub value_fn_learning_rate: Option<f64>,

    #[clap(long, help_heading = Some("AGENT VALUE FN OPTIMIZER OPTIONS"))]
    /// Agent value function optimizer momentum
    pub value_fn_momentum: Option<f64>,

    #[clap(long, help_heading = Some("AGENT VALUE FN OPTIMIZER OPTIONS"))]
    /// Agent value function optimizer weight decay (L2 regularization)
    pub value_fn_weight_decay: Option<f64>,

    #[clap(long, help_heading = Some("AGENT VALUE FN OPTIMIZER OPTIONS"))]
    /// Number of value function training iterations per epoch
    pub value_fn_train_iters: Option<u64>,

    // Conjugate Gradient Optimizer options
    #[clap(long, help_heading = Some("CONJUGATE GRADIENT OPTIMIZER OPTIONS"))]
    /// Number of conjugate gradient iterations used to solve for the step direction
    pub cg_iterations: Option<u64>,

    #[clap(long, help_heading = Some("CONJUGATE GRADIENT OPTIMIZER OPTIONS"))]
    /// Maximum number of iterations for backtracking line search
    pub cg_max_backtracks: Option<u64>,

    #[clap(long, help_heading = Some("CONJUGATE GRADIENT OPTIMIZER OPTIONS"))]
    /// Multiplicative scale factor applied on each line search backtrack iteration
    pub cg_backtrack_ratio: Option<f64>,

    #[clap(long, help_heading = Some("CONJUGATE GRADIENT OPTIMIZER OPTIONS"))]
    /// A small regularization coefficient for stability when inverting the Hessian
    pub cg_hpv_reg_coeff: Option<f64>,

    #[clap(long, help_heading = Some("CONJUGATE GRADIENT OPTIMIZER OPTIONS"))]
    /// Whether to accept the conjugate gradient descent step if the distance condition is violated
    pub cg_accept_violation: Option<bool>,

    // Simulation options
    #[clap(long, default_value = "1", help_heading = Some("SIMULATION OPTIONS"))]
    /// Random seed for the experiment
    pub seed: u64,

    #[clap(long, help_heading = Some("SIMULATION OPTIONS"))]
    /// Maximum number of experiment steps
    pub max_steps: Option<u64>,
}

impl Options {
    pub const fn value_fn_view(&self) -> ValueFnView {
        ValueFnView(self)
    }
}

impl OptimizerOptions for Options {
    fn type_(&self) -> Option<OptimizerType> {
        self.optimizer
    }
    fn learning_rate(&self) -> Option<f64> {
        self.learning_rate
    }
    fn momentum(&self) -> Option<f64> {
        self.momentum
    }
    fn weight_decay(&self) -> Option<f64> {
        self.weight_decay
    }
}

/// An option view that exposes value function options.
pub struct ValueFnView<'a>(&'a Options);

impl<'a> OptimizerOptions for ValueFnView<'a> {
    fn type_(&self) -> Option<OptimizerType> {
        self.0.value_fn_optimizer
    }
    fn learning_rate(&self) -> Option<f64> {
        self.0.value_fn_learning_rate
    }
    fn momentum(&self) -> Option<f64> {
        self.0.value_fn_momentum
    }
    fn weight_decay(&self) -> Option<f64> {
        self.0.value_fn_weight_decay
    }
}
