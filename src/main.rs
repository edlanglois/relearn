use clap::Clap;
use rust_rl::logging::CLILogger;
use rust_rl::{cli::Options, AgentDef, EnvDef};
use std::convert::From;
use std::error::Error;
use std::time::Duration;

fn main() -> Result<(), Box<dyn Error>> {
    let opts: Options = Options::parse();
    println!("{:#?}", opts);
    let env_def = EnvDef::from(&opts);
    println!("Environment:\n{:#?}", env_def);
    let agent_def = AgentDef::from(&opts);
    println!("Agent:\n{:#?}", agent_def);

    let logger = CLILogger::new(Duration::from_millis(1000), true);
    let mut simulation =
        env_def.build_simulation(&agent_def, opts.seed, opts.seed + 1, logger, ())?;
    simulation.run();
    Ok(())
}
