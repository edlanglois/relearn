use clap::{ArgEnum, Parser};
use rand::SeedableRng;
use relearn::agents::{
    Actor, BetaThompsonSamplingAgentConfig, BuildAgent, RandomAgentConfig, ResettingMetaAgent,
    TabularQLearningAgentConfig, UCB1AgentConfig,
};
use relearn::envs::{
    BuildEnv, Environment, MetaEnv, MetaObservationSpace, StructuredEnvironment,
    UniformBernoulliBandits,
};
use relearn::simulation::{SimulatorSteps, StepsIter, StepsSummary};
use relearn::spaces::{NonEmptySpace, Space};
use relearn::Prng;

#[derive(Parser, Debug, Clone, PartialEq)]
#[clap(
    name = "rl2-bandits-eval",
    author,
    about = "Evaluate multi-armed bandit algorithms as in the RL2 paper"
)]
pub struct Args {
    /// Number of bandit arms
    #[clap(short = 'n', long, default_value_t = 10)]
    pub num_arms: usize,

    /// Number of episodes per trial
    #[clap(short = 'k', long, default_value_t = 100)]
    pub num_episodes: u64,

    /// Number of evaluation trials
    #[clap(long, default_value_t = 1000)]
    pub num_trials: usize,

    /// Agent type
    #[clap(short, long, arg_enum, default_value_t = AgentType::UCB1)]
    pub agent: AgentType,

    /// Environment random seed
    #[clap(long)]
    pub env_seed: Option<u64>,

    /// Agent random seed
    #[clap(long)]
    pub agent_seed: Option<u64>,

    /// Enable verbose output
    #[clap(short, long)]
    pub verbose: bool,
}

/// Agent type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ArgEnum)]
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
    SimulatorSteps::new(env, actor, rng_env, rng_actor, ())
        .take_episodes(num_episodes)
        .collect()
}

fn new_prng(seed: Option<u64>) -> Prng {
    match seed {
        None => Prng::from_entropy(),
        Some(s) => Prng::seed_from_u64(s),
    }
}

fn main() {
    let args = Args::parse();
    if args.verbose {
        println!("{args:#?}");
    }

    let env_config = MetaEnv {
        env_distribution: UniformBernoulliBandits::new(args.num_arms),
        episodes_per_trial: args.num_episodes,
    };
    if args.verbose {
        println!("{env_config:#?}");
    }

    let mut rng_env = new_prng(args.env_seed);
    let rng_agent = new_prng(args.agent_seed);

    let env = env_config.build_env(&mut rng_env).unwrap();

    let summary = args
        .agent
        .evaluate(env, args.num_trials, rng_env, rng_agent);
    println!("{summary:.3}");
}
