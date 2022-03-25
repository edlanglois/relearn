use clap::{ArgEnum, Parser, Subcommand};
use rand::{Rng, SeedableRng};
use relearn::agents::{
    Actor, BetaThompsonSamplingAgentConfig, BuildAgent, RandomAgentConfig, ResettingMetaAgent,
    TabularQLearningAgentConfig, UCB1AgentConfig,
};
use relearn::envs::{
    BuildEnv, Environment, MetaEnv, MetaObservationSpace, StructuredEnvironment,
    UniformBernoulliBandits,
};
use relearn::logging::{ByCounter, DisplayLogger, StatsLogger, TensorBoardLogger};
use relearn::simulation::{train_parallel, TrainParallelConfig};
use relearn::simulation::{Steps, StepsIter, StepsSummary};
use relearn::spaces::{IndexSpace, NonEmptySpace, SingletonSpace, Space};
use relearn::torch::{
    agents::{
        critic::{Gae, GaeConfig},
        learning_critic::{GradOpt, GradOptConfig, GradOptRule},
        learning_policy::{Trpo, TrpoConfig, TrpoRule},
        ActorCriticAgent, ActorCriticConfig,
    },
    initializers::{Initializer, VarianceScale},
    modules::{Activation, Chain, ChainConfig, Gru, GruConfig, Linear, LinearConfig},
    optimizers::{AdamConfig, ConjugateGradientOptimizerConfig},
};
use relearn::Prng;
use serde::Serialize;
use std::fmt;
use std::num::ParseIntError;
use std::path::PathBuf;
use std::str::FromStr;
use thiserror::Error;

#[derive(Parser, Debug, Clone, PartialEq)]
#[clap(
    name = "rl2-bandits",
    author,
    about = "Train and evaluate multi-armed bandit algorithms as in the RL2 paper"
)]
pub struct Args {
    /// Experiment Action
    #[clap(subcommand)]
    action: Action,

    /// Number of bandit arms
    #[clap(short = 'k', long, default_value_t = 10)]
    pub num_arms: usize,

    /// Number of episodes per trial
    #[clap(short = 'n', long, default_value_t = 100)]
    pub num_episodes: u64,

    /// Number of evaluation trials
    #[clap(long, default_value_t = 1000)]
    pub num_trials: usize,

    /// Environment random seed
    #[clap(long)]
    pub env_seed: Option<u64>,

    /// Agent random seed
    #[clap(long)]
    pub agent_seed: Option<u64>,

    /// Suppress status output
    #[clap(short, long)]
    pub silent: bool,

    /// Output format
    #[clap(short, long, arg_enum, default_value_t = OutputFormat::Human)]
    pub output: OutputFormat,
}

impl Args {
    fn config(&self) -> Config {
        match &self.action {
            Action::Train {
                batch_size,
                device,
                output_dir,
            } => Config::Train(TrainConfig {
                num_arms: self.num_arms,
                num_episodes: self.num_episodes,
                num_trials: self.num_trials,
                env_seed: self.env_seed.unwrap_or_else(|| rand::thread_rng().gen()),
                agent_seed: self.agent_seed.unwrap_or_else(|| rand::thread_rng().gen()),
                batch_size: *batch_size,
                device: *device,
                num_workers: num_cpus::get(),
                output_dir: output_dir.clone().into(),
            }),
            Action::Eval { agent } => Config::Eval(EvalConfig {
                num_arms: self.num_arms,
                num_episodes: self.num_episodes,
                num_trials: self.num_trials,
                agent: *agent,
                env_seed: self.env_seed.unwrap_or_else(|| rand::thread_rng().gen()),
                agent_seed: self.agent_seed.unwrap_or_else(|| rand::thread_rng().gen()),
            }),
        }
    }
}

fn default_output_dir() -> DisplayPathBuf {
    let mut output_dir = xdg::BaseDirectories::with_prefix("relearn")
        .unwrap()
        .get_data_file("rl2-bandits");
    output_dir.push(chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string());
    output_dir.into()
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct DisplayPathBuf(pub PathBuf);
impl fmt::Display for DisplayPathBuf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0.display(), f)
    }
}
impl From<PathBuf> for DisplayPathBuf {
    fn from(path: PathBuf) -> Self {
        Self(path)
    }
}
impl From<DisplayPathBuf> for PathBuf {
    fn from(dpath: DisplayPathBuf) -> Self {
        dpath.0
    }
}
impl FromStr for DisplayPathBuf {
    type Err = <PathBuf as FromStr>::Err;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(PathBuf::from_str(s)?.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Subcommand)]
pub enum Action {
    /// Train the RL2 policy
    Train {
        /// Batch size for policy updates
        #[clap(long, default_value_t = 250_000)]
        batch_size: usize,

        /// Device on which to perform model updates
        #[clap(long, default_value_t = Device::Cpu)]
        device: Device,

        /// Output directory for tensorboard logs
        #[clap(long, default_value_t = default_output_dir())]
        output_dir: DisplayPathBuf,
    },
    /// Evaluate an algorithm
    Eval {
        /// Agent type
        #[clap(short, long, arg_enum, default_value_t = AgentType::UCB1)]
        agent: AgentType,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
pub enum OutputFormat {
    Human,
    Json,
}

/// Agent type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum, Serialize)]
pub enum AgentType {
    /// Uniform random
    Random,
    /// Epsilon-greedy
    EpsGreedy,
    /// Greedy
    Greedy,
    /// Thompson Sampling
    TS,
    /// Optimistic Thompson Sampling
    OTS,
    /// Upper Confidence Bound Alg 1
    UCB1,
}

impl AgentType {
    fn evaluate(
        self,
        meta_env: MetaEnv<UniformBernoulliBandits>,
        num_trials: usize,
        rng_env: Prng,
        rng_agent: Prng,
    ) -> StepsSummary {
        macro_rules! eval_resetting {
            ($config:expr) => {
                eval_resetting_meta(meta_env, $config, num_trials, rng_env, rng_agent)
            };
        }

        match self {
            AgentType::Random => {
                eval_resetting!(RandomAgentConfig)
            }
            AgentType::EpsGreedy => {
                eval_resetting!(TabularQLearningAgentConfig {
                    exploration_rate: 0.2, // paper does not give the value they used
                    initial_action_count: 2,
                    initial_action_value: 0.5,
                })
            }
            AgentType::Greedy => eval_resetting!(TabularQLearningAgentConfig {
                exploration_rate: 0.0,
                initial_action_count: 2,
                initial_action_value: 0.5,
            }),
            AgentType::TS => eval_resetting!(BetaThompsonSamplingAgentConfig::default()),
            AgentType::OTS => eval_resetting!(BetaThompsonSamplingAgentConfig {
                num_samples: 10, // paper does not give the value they used
            }),
            AgentType::UCB1 => eval_resetting!(UCB1AgentConfig {
                exploration_rate: 0.2, // paper does not give the value they used
            }),
        }
    }
}

fn eval_resetting_meta<E, TC, OS, AS>(
    meta_env: E,
    actor_config: TC,
    num_trials: usize,
    rng_env: Prng,
    rng_actor: Prng,
) -> StepsSummary
where
    E: StructuredEnvironment<ObservationSpace = MetaObservationSpace<OS, AS>, ActionSpace = AS>,
    TC: BuildAgent<OS, AS>,
    OS: Space + Clone,
    AS: NonEmptySpace + Clone,
{
    let meta_actor = ResettingMetaAgent::from_meta_env(actor_config, &meta_env);
    eval(meta_env, meta_actor, num_trials, rng_env, rng_actor)
}

fn eval<E, T>(env: E, actor: T, num_episodes: usize, rng_env: Prng, rng_actor: Prng) -> StepsSummary
where
    E: Environment,
    T: Actor<E::Observation, E::Action>,
{
    Steps::new(env, actor, rng_env, rng_actor, ())
        .take_episodes(num_episodes)
        .collect()
}

enum Config {
    Train(TrainConfig),
    Eval(EvalConfig),
}

fn make_env(
    num_arms: usize,
    num_episodes: u64,
    rng: &mut Prng,
    verbose: bool,
) -> MetaEnv<UniformBernoulliBandits> {
    let env_config = MetaEnv {
        env_distribution: UniformBernoulliBandits::new(num_arms),
        episodes_per_trial: num_episodes,
    };
    if verbose {
        println!("{env_config:#?}\n");
    }

    env_config.build_env(rng).unwrap()
}

/// Serializable reimplementation of [`tch::Device`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

/// Parse "cpu" or "cuda:#"
impl FromStr for Device {
    type Err = ParseDeviceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (device_str, index_str) = match s.split_once(':') {
            Some((dev, index)) => (dev, Some(index)),
            None => (s, None),
        };
        if device_str.eq_ignore_ascii_case("cpu") {
            if index_str.is_some() {
                Err(ParseDeviceError::UnexpectedIndex)
            } else {
                Ok(Device::Cpu)
            }
        } else if device_str.eq_ignore_ascii_case("cuda") {
            let index = index_str.map(str::parse).unwrap_or(Ok(0))?;
            Ok(Device::Cuda(index))
        } else {
            Err(ParseDeviceError::InvalidDevice)
        }
    }
}

/// Error parsing a [`Device`]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParseDeviceError {
    #[error("invalid device name")]
    InvalidDevice,
    #[error("invalid device index")]
    InvalidIndex(#[from] ParseIntError),
    #[error("unexpected device index")]
    UnexpectedIndex,
}

impl From<Device> for tch::Device {
    fn from(d: Device) -> Self {
        match d {
            Device::Cpu => Self::Cpu,
            Device::Cuda(index) => Self::Cuda(index),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(index) => write!(f, "cuda:{}", index),
        }
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

type Model = Chain<Gru, Linear>;
type Agent = ActorCriticAgent<
    MetaObservationSpace<SingletonSpace, IndexSpace>,
    IndexSpace,
    Trpo<Model>,
    GradOpt<Gae<Model>>,
>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrainConfig {
    pub num_arms: usize,
    pub num_episodes: u64,
    pub num_trials: usize,
    pub env_seed: u64,
    pub agent_seed: u64,

    pub batch_size: usize,
    pub device: Device,
    pub num_workers: usize,
    pub output_dir: PathBuf,
}

impl TrainConfig {
    fn train_policy(&self, verbose: bool) -> Agent {
        let mut rng_env = Prng::seed_from_u64(self.env_seed);
        let mut rng_agent = Prng::seed_from_u64(self.agent_seed);
        let env = make_env(self.num_arms, self.num_episodes, &mut rng_env, verbose);

        // Configuration for the policy and baseline networks
        // GRU followed by fully-connected
        let model_config = ChainConfig {
            first_config: GruConfig {
                num_layers: 1,
                input_weights_init: Initializer::Uniform(VarianceScale::FanAvg),
                hidden_weights_init: Initializer::Orthogonal,
                bias_init: Some(Initializer::Zeros),
                ..GruConfig::default()
            },
            second_config: LinearConfig::default(),
            hidden_dim: 128,
            // They might have also used Relu within the GRU. This only controls the activation
            // between the hidden weights and the linear output layer.
            // I'm not sure that it is possible to change the activation used by GRU in torch.
            // NOTE: Could also try with no activation here if the GRU nonlinearity is sufficient.
            activation: Activation::Relu,
        };
        let policy_config = TrpoConfig {
            module_config: model_config,
            // RL2 paper has "Policy Iters: Up to 1000".
            // I think this refers to number of update periods,
            // not the number of CG iterations when solving for for the descent direction.
            // They do not seem to specify any CG optimizer parameters.
            optimizer_config: ConjugateGradientOptimizerConfig::default(),
            update_rule_config: TrpoRule {
                // RL2 paper mentions "Mean KL" as a parameter that they set to 0.01.
                // I do not know of such a parameter for TRPO.
                // The TRPO paper has a max KL step size parameter that they set to 0.01 so
                // I assume that either this is what was meant by the RL2 paper or
                // that their strategy was not too different.
                max_policy_step_kl: 0.01,
            },
        };
        // Note: I think they used CG optimization here as in the GAE paper
        // https://arxiv.org/pdf/1506.02438.pdf, not regular gradient descent.
        // TODO: Implement and use CG for critic updates.
        let critic_config = GradOptConfig {
            module_config: GaeConfig {
                gamma: 0.99,
                lambda: 0.3,
                value_fn_config: model_config,
            },
            optimizer_config: AdamConfig::default(),
            update_rule_config: GradOptRule {
                optimizer_iters: 50,
            },
        };
        let agent_config = ActorCriticConfig {
            policy_config,
            critic_config,
            min_batch_steps: self.batch_size,
            device: self.device.into(),
        };
        let mut agent = agent_config.build_agent(&env, &mut rng_agent).unwrap();

        let training_config = TrainParallelConfig {
            num_periods: self.num_trials,
            num_threads: self.num_workers,
            min_workers_steps: 10_000,
        };

        let log_chunker = ByCounter::of_path(["agent_update", "count"], 1);
        let tb_logger = TensorBoardLogger::new(log_chunker.clone(), &self.output_dir);
        if verbose {
            println!("Output Dir: {}", self.output_dir.display());
        }
        let mut logger: Box<dyn StatsLogger> = if verbose {
            Box::new((DisplayLogger::new(log_chunker), tb_logger))
        } else {
            Box::new(tb_logger)
        };

        train_parallel(
            &mut agent,
            &env,
            &training_config,
            &mut rng_env,
            &mut rng_agent,
            logger.as_mut(),
        );

        agent
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize)]
pub struct EvalConfig {
    pub num_arms: usize,
    pub num_episodes: u64,
    pub num_trials: usize,
    pub agent: AgentType,
    pub env_seed: u64,
    pub agent_seed: u64,
}

impl EvalConfig {
    fn run_experiment(&self, verbose: bool) -> EvalResults {
        let mut rng_env = Prng::seed_from_u64(self.env_seed);
        let env = make_env(self.num_arms, self.num_episodes, &mut rng_env, verbose);

        let rng_agent = Prng::seed_from_u64(self.agent_seed);
        let summary = self
            .agent
            .evaluate(env, self.num_trials, rng_env, rng_agent);

        if verbose {
            println!("{summary:.3}\n");
        }
        EvalResults {
            trial_reward_mean: summary.episode_reward.mean().unwrap_or(f64::NAN),
            trial_reward_stddev: summary.episode_reward.stddev().unwrap_or(f64::NAN),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvalResults {
    pub trial_reward_mean: f64,
    pub trial_reward_stddev: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvalData {
    pub config: EvalConfig,
    pub results: EvalResults,
}

fn main() {
    let args = Args::parse();
    match args.config() {
        Config::Train(config) => {
            let _ = config.train_policy(!args.silent);
        }
        Config::Eval(config) => {
            let results = config.run_experiment(!args.silent);

            let data = EvalData { config, results };
            match args.output {
                OutputFormat::Human => {
                    println!("# Config\n{:#?}\n", data.config);
                    println!("# Results");
                    println!("trial_reward_mean: {:.3}", data.results.trial_reward_mean);
                    println!(
                        "trial_reward_stddev: {:.3}",
                        data.results.trial_reward_stddev
                    );
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string(&data).unwrap());
                }
            }
        }
    };
}
