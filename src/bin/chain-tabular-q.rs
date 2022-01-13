use rand::{Rng, SeedableRng};
use relearn::agents::{ActorMode, Agent, BuildAgent, TabularQLearningAgentConfig};
use relearn::envs::{BuildEnv, Chain, Environment};
use relearn::logging::CLILoggerConfig;
use relearn::simulation::{train_parallel, SimulationSummary, TrainParallelConfig};
use relearn::Prng;

fn main() {
    let env_config = Chain::default();
    let agent_config = TabularQLearningAgentConfig::default();
    let logger_config = CLILoggerConfig::default();
    let training_config = TrainParallelConfig {
        num_periods: 10,
        num_threads: num_cpus::get(),
        min_workers_steps: 10_000,
    };

    let mut rng = Prng::seed_from_u64(0);
    let env = env_config.build_env(&mut rng).unwrap();
    let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();
    let mut logger = logger_config.build_logger();

    {
        let summary = SimulationSummary::from_steps(
            env.run(&agent.actor(ActorMode::Evaluation), rng.gen(), ())
                .take(10_000),
        );
        println!("Initial Stats\n{}", summary);
    }

    train_parallel(
        &mut agent,
        &env,
        &training_config,
        &mut Prng::from_rng(&mut rng).unwrap(),
        &mut rng,
        &mut logger,
    );

    let summary = SimulationSummary::from_steps(
        env.run(&agent.actor(ActorMode::Evaluation), rng.gen(), ())
            .take(10_000),
    );
    println!("\nFinal Stats\n{}", summary);
}
