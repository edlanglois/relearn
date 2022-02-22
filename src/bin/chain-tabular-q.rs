use rand::SeedableRng;
use relearn::agents::{ActorMode, Agent, BuildAgent, TabularQLearningAgentConfig};
use relearn::envs::{BuildEnv, Chain, Environment};
use relearn::logging::DisplayLogger;
use relearn::simulation::{train_parallel, SimSeed, StepsSummary, TrainParallelConfig};
use relearn::Prng;

fn main() {
    let env_config = Chain::default();
    // let env_config = Chain::default().with_step_limit(100);
    let agent_config = TabularQLearningAgentConfig::default();
    let training_config = TrainParallelConfig {
        num_periods: 10,
        num_threads: num_cpus::get(),
        min_workers_steps: 10_000,
    };

    let mut rng = Prng::seed_from_u64(0);
    let env = env_config.build_env(&mut rng).unwrap();
    let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();
    let mut logger = DisplayLogger::default();

    {
        let summary: StepsSummary = env
            .run(&agent.actor(ActorMode::Evaluation), SimSeed::Root(0), ())
            .take(10_000)
            .collect();
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

    let summary: StepsSummary = env
        .run(&agent.actor(ActorMode::Evaluation), SimSeed::Root(0), ())
        .take(10_000)
        .collect();
    println!("\nFinal Stats\n{}", summary);
}
