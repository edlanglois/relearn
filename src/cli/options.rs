//! Command-line options
use super::agent::{AgentType, AgentWrapperType, ConcreteAgentType};
use super::critic::CriticType;
use super::env::{BanditArmPrior, EnvType};
use super::optimizer::{OptimizerOptions, OptimizerType};
use super::seq_mod::SeqModType;
use crate::torch::Activation;
use clap::{crate_authors, crate_description, crate_version, ArgEnum, Parser};
use once_cell::sync::OnceCell;
use std::{error::Error, fmt, marker::PhantomData, str::FromStr};
use tch::Device;

#[derive(Debug, Clone, PartialEq, Parser)]
#[clap(
    version = crate_version!(),
    author = crate_authors!(),
    about = crate_description!(),
    after_help = "Most options only apply for some environments/agents and are ignored otherwise.",
)]
pub struct Options {
    // Environment options
    #[clap(short, long, arg_enum, help_heading = Some("ENVIRONMENT OPTIONS"))]
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

    #[clap(long, help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// Number of inner episodes per trial in meta environments.
    pub episodes_per_trial: Option<usize>,

    #[clap(long, help_heading = Some("ENVIRONMENT OPTIONS"))]
    /// Max steps per episode; for environments with a step limit
    pub max_steps_per_episode: Option<u64>,

    // Agent args
    /// Agent type
    #[clap(short, long, long_about = agent_long_about(), help_heading = Some("AGENT OPTIONS"))]
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

    #[clap(long, help_heading = Some("AGENT OPTIONS"))]
    /// Device on which the agent parameters are stored [possible values: cpu, cuda, cuda:N]
    pub device: Option<DeviceOpt>,

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

    // Critic options
    #[clap(long, arg_enum, help_heading = Some("AGENT CRITIC OPTIONS"))]
    /// Agent critic type
    pub critic: Option<CriticType>,

    #[clap(long, help_heading = Some("AGENT CRITIC OPTIONS"))]
    /// Maximum discount factor used by GAE
    pub gae_discount_factor: Option<f64>,

    #[clap(long, help_heading = Some("AGENT CRITIC OPTIONS"))]
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
    /// Number of policy epochs per update period
    pub policy_epochs: Option<u64>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Clip distance for proximal policy optimization
    pub ppo_clip_distance: Option<f64>,

    #[clap(long, help_heading = Some("AGENT OPTIMIZER OPTIONS"))]
    /// Max policy KL divergence per update step; for trust-region methods.
    pub max_policy_step_kl: Option<f64>,

    #[clap(long, arg_enum, help_heading = Some("AGENT CRITIC OPTIMIZER OPTIONS"))]
    /// Agent critic optimizer type
    pub critic_optimizer: Option<OptimizerType>,

    #[clap(long, help_heading = Some("AGENT CRITIC OPTIMIZER OPTIONS"))]
    /// Agent critic optimizer learning rate
    pub critic_learning_rate: Option<f64>,

    #[clap(long, help_heading = Some("AGENT CRITIC OPTIMIZER OPTIONS"))]
    /// Agent critic optimizer momentum
    pub critic_momentum: Option<f64>,

    #[clap(long, help_heading = Some("AGENT CRITIC OPTIMIZER OPTIONS"))]
    /// Agent critic optimizer weight decay (L2 regularization)
    pub critic_weight_decay: Option<f64>,

    #[clap(long, help_heading = Some("AGENT CRITIC OPTIMIZER OPTIONS"))]
    /// Number of critic optimizer step iterations per epoch
    pub critic_opt_iters: Option<u64>,

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
    /// Maximum number of experiment steps (total steps when running parallel simulations)"
    pub max_steps: Option<u64>,

    #[clap(long, help_heading = Some("SIMULATION OPTIONS"))]
    /// Maximum number of experiment episodes (total episodes when running parallel simulations)"
    pub max_episodes: Option<u64>,

    #[clap(short, long, visible_alias="parallel", default_missing_value="0",
           help_heading = Some("SIMULATION OPTIONS"))]
    /// Number of parallel simulation threads. An additional thread runs the agent manager.
    ///
    /// Passing this option without an argument (or 0) defaults to the number of CPU cores.
    pub parallel_threads: Option<usize>,

    #[clap(long, help_heading = Some("SIMULATION OPTIONS"))]
    /// Log display period in seconds
    pub display_period: Option<u64>,
}

/// Stores the agent argument help message.
///
/// The `(long_)about` argument method takes a &str.
/// We construct a help message using format! which
///     1) involves a runtime function and
///     2) returns a String.
/// Since we are using the derive method of construcing `Options`,
/// there is no local context in which to store the formatted string
/// so that a reference can be given to `(long_)about`.
///
/// Instead, we use a static string and set its value at runtime.
static AGENT_ABOUT: OnceCell<String> = OnceCell::new();

/// Agent argument long about message
fn agent_long_about() -> &'static str {
    AGENT_ABOUT.get_or_init(|| {
        format!(
            "Agent type. Format: [<wrapper>:]*<base-agent>

base-agent: {}
wrapper: {}
",
            ArgEnumVariantDisplay::<ConcreteAgentType>::new(),
            ArgEnumVariantDisplay::<AgentWrapperType>::new()
        )
    })
}

/// Wrapper for displaying the variant names of an `ArgEnum` type
struct ArgEnumVariantDisplay<T>(PhantomData<*const T>);
impl<T> ArgEnumVariantDisplay<T> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}
impl<T: ArgEnum> fmt::Display for ArgEnumVariantDisplay<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for v in T::value_variants() {
            if let Some(value) = v.to_arg_value() {
                if !first {
                    write!(f, ", ")?
                }
                first = false;
                write!(f, "{}", value.get_name())?;
            }
        }
        Ok(())
    }
}

impl Options {
    pub const fn critic_view(&self) -> CriticView {
        CriticView(self)
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

/// An option view that exposes critic options.
#[derive(Debug, Clone, PartialEq)]
pub struct CriticView<'a>(&'a Options);

impl<'a> OptimizerOptions for CriticView<'a> {
    fn type_(&self) -> Option<OptimizerType> {
        self.0.critic_optimizer
    }
    fn learning_rate(&self) -> Option<f64> {
        self.0.critic_learning_rate
    }
    fn momentum(&self) -> Option<f64> {
        self.0.critic_momentum
    }
    fn weight_decay(&self) -> Option<f64> {
        self.0.critic_weight_decay
    }
}

/// Device wrapper implementing [`FromStr`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceOpt(Device);

impl DeviceOpt {
    pub const fn new(device: Device) -> Self {
        Self(device)
    }
}

impl FromStr for DeviceOpt {
    type Err = DeviceParseError;

    fn from_str(s: &str) -> Result<Self, DeviceParseError> {
        Ok(Self(parse_device(s)?))
    }
}

impl From<Device> for DeviceOpt {
    fn from(device: Device) -> Self {
        Self(device)
    }
}

impl From<DeviceOpt> for Device {
    fn from(device_opt: DeviceOpt) -> Self {
        device_opt.0
    }
}

/// Parse from "cpu", "cuda", or "cuda:N" for non-negative integer N.
///
/// The match is ASCII case insensitive.
fn parse_device(s: &str) -> Result<Device, DeviceParseError> {
    if s.eq_ignore_ascii_case("cpu") {
        return Ok(Device::Cpu);
    }

    let (name, index) = if let Some((name, index_str)) = s.split_once(':') {
        let index = index_str.parse().map_err(|_| DeviceParseError)?;
        (name, index)
    } else {
        (s, 0)
    };
    if name.eq_ignore_ascii_case("cuda") {
        return Ok(Device::Cuda(index));
    }
    Err(DeviceParseError)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct DeviceParseError;

impl fmt::Display for DeviceParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Invalid device string. \
               Options: \"cpu\", \"cuda\", or \"cuda:[INDEX]\" (case insensitive)",
        )
    }
}

impl Error for DeviceParseError {}
