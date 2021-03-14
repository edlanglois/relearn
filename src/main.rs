use rust_rl::loggers::CLILogger;
use rust_rl::simulator::{AgentDef, EnvDef};
use std::error::Error;
use std::time::Duration;

fn main() -> Result<(), Box<dyn Error>> {
    let env_def = EnvDef::SimpleBernoulliBandit;
    let agent_def = AgentDef::TabularQLearning {
        exploration_rate: 0.1,
    };
    let seed = 1;
    let num_steps = 10_000_000;

    println!("Environment: {:?}", env_def);
    println!("Agent: {:?}", agent_def);

    let logger = CLILogger::new(Duration::from_millis(1000), true);
    let mut simulator = env_def.make_simulator(agent_def, seed, logger)?;
    simulator.run(Some(num_steps));
    Ok(())
}
