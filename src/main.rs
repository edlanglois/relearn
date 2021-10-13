use clap::Clap;
use relearn::cli::Options;
use relearn::defs::{boxed_multithread_simulator, boxed_serial_simulator, HookDef, HooksDef};
use relearn::logging::CLILogger;
use relearn::simulation::{hooks::StepLoggerConfig, MultithreadSimulatorConfig};
use relearn::{AgentDef, EnvDef, MultithreadAgentDef};
use std::convert::From;
use std::error::Error;
use std::time::Duration;

fn run_serial(opts: &Options, env_def: EnvDef, hook_def: HooksDef) -> Result<(), Box<dyn Error>> {
    let agent_def = Option::<AgentDef>::from(opts)
        .ok_or_else(|| format!("Agent {} is not single-threaded", opts.agent))?;
    println!("Agent:\n{:#?}", agent_def);

    let env_seed = opts.seed;
    let agent_seed = opts.seed.wrapping_add(1);
    let mut simulation = boxed_serial_simulator(env_def, agent_def, hook_def);
    let mut logger = CLILogger::new(Duration::from_millis(1000));
    simulation
        .run_simulation(env_seed, agent_seed, &mut logger)
        .expect("Simulation failed");
    Ok(())
}

fn run_multithread(
    opts: &Options,
    env_def: EnvDef,
    hook_def: HooksDef,
    mut num_threads: usize,
) -> Result<(), Box<dyn Error>> {
    let agent_def = Option::<MultithreadAgentDef>::from(opts)
        .ok_or_else(|| format!("Agent {} is not multi-threaded", opts.agent))?;
    println!("Agent:\n{:#?}", agent_def);

    if num_threads == 0 {
        num_threads = num_cpus::get();
    }
    let sim_config = MultithreadSimulatorConfig {
        num_workers: num_threads,
    };
    println!("Simulation:\n{:#?}", sim_config);
    println!(
        "Starting a parallel run with {} simulation threads.",
        num_threads
    );

    let env_seed = opts.seed;
    let agent_seed = opts.seed.wrapping_add(1);
    let mut simulation = boxed_multithread_simulator(sim_config, env_def, agent_def, hook_def);
    let mut logger = CLILogger::new(Duration::from_millis(1000));
    simulation
        .run_simulation(env_seed, agent_seed, &mut logger)
        .expect("Simulation failed");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let opts: Options = Options::parse();
    println!("{:#?}", opts);
    let env_def = EnvDef::from(&opts);
    println!("Environment:\n{:#?}", env_def);

    // TODO: Load from opts
    let hook_def = HooksDef::new(vec![HookDef::StepLogger(StepLoggerConfig)]);

    if let Some(num_threads) = opts.parallel_threads {
        run_multithread(&opts, env_def, hook_def, num_threads)
    } else {
        run_serial(&opts, env_def, hook_def)
    }
}
