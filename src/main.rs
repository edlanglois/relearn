use clap::{crate_authors, crate_description, crate_version, Clap};
use rust_rl::logging::CLILogger;
use rust_rl::simulation::{AgentDef, EnvDef};
use std::convert::From;
use std::error::Error;
use std::time::Duration;

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
    environment: Env,

    #[clap(long)]
    /// Number of states in the environment; when configurable
    num_states: Option<u32>,

    #[clap(long)]
    /// Number of actions in the environment; when configurable
    num_actions: Option<u32>,

    #[clap(long)]
    /// Environment discount factor; when configurable
    discount_factor: Option<f64>,

    // Agent args
    #[clap(arg_enum)]
    /// Agent name
    agent: Agent,

    #[clap(long, default_value = "0.2")]
    /// Agent exploration rate
    exploration_rate: f64,

    #[clap(long, default_value = "1")]
    num_samples: usize,

    // Experiment args
    #[clap(long)]
    /// Maximum number of experiment steps
    max_steps: Option<u64>,
}

#[derive(Clap, Debug)]
pub enum Env {
    SimpleBernoulliBandit,
    BernoulliBandit,
    DeterministicBandit,
    Chain,
}

impl From<&Opts> for EnvDef {
    fn from(opts: &Opts) -> Self {
        match opts.environment {
            Env::SimpleBernoulliBandit => EnvDef::SimpleBernoulliBandit,
            Env::BernoulliBandit => EnvDef::BernoulliBandit {
                num_arms: opts.num_actions.unwrap_or(2),
            },
            Env::DeterministicBandit => EnvDef::DeterministicBandit {
                num_arms: opts.num_actions.unwrap_or(2),
            },
            Env::Chain => EnvDef::Chain {
                num_states: opts.num_states,
                discount_factor: opts.discount_factor,
            },
        }
    }
}

#[derive(Clap, Debug)]
pub enum Agent {
    Random,
    TabularQLearning,
    BetaThompsonSampling,
}

impl From<&Opts> for AgentDef {
    fn from(opts: &Opts) -> Self {
        match opts.agent {
            Agent::Random => AgentDef::Random,
            Agent::TabularQLearning => AgentDef::TabularQLearning {
                exploration_rate: opts.exploration_rate,
            },
            Agent::BetaThompsonSampling => AgentDef::BetaThompsonSampling {
                num_samples: opts.num_samples,
            },
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let opts: Opts = Opts::parse();
    println!("{:?}", opts);
    let env_def = EnvDef::from(&opts);
    println!("Environment: {:?}", env_def);
    let agent_def = AgentDef::from(&opts);
    println!("Agent: {:?}", agent_def);

    let logger = CLILogger::new(Duration::from_millis(1000), true);
    let mut simulation = env_def.make_simulation(agent_def, opts.seed, logger)?;
    simulation.run(opts.max_steps);
    Ok(())
}
