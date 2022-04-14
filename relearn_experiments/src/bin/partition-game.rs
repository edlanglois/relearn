use rand::SeedableRng;
use relearn::agents::{ActorMode, Agent, BuildAgent};
use relearn::envs::{BuildEnv, Environment, PartitionGame};
use relearn::logging::DisplayLogger;
use relearn::simulation::{train_parallel, SimSeed, StepsIter, TrainParallelConfig};
use relearn::torch::agents::{
    critic::GaeConfig, learning_critic::GradOptConfig, learning_policy::TrpoConfig,
    ActorCriticConfig,
};
use relearn::torch::modules::GruMlpConfig;
use relearn::Prng;
use tch::Device;

fn main() {
    let env_config = PartitionGame::default().with_visible_step_limit(1000);

    let agent_config: ActorCriticConfig<
        TrpoConfig<GruMlpConfig>,
        GradOptConfig<GaeConfig<GruMlpConfig>>,
    > = ActorCriticConfig {
        device: Device::Cuda(0),
        ..Default::default()
    };
    let training_config = TrainParallelConfig {
        num_periods: 1000,
        num_threads: num_cpus::get(),
        min_worker_steps: 5_000,
    };
    let mut rng = Prng::seed_from_u64(0);
    let env = env_config.build_env(&mut rng).unwrap();
    let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();
    let mut logger: DisplayLogger = DisplayLogger::default();

    {
        let summary = env
            .run(&agent.actor(ActorMode::Evaluation), SimSeed::Root(0), ())
            .take(10_000)
            .summarize();
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

    let summary = env
        .run(&agent.actor(ActorMode::Evaluation), SimSeed::Root(0), ())
        .take(10_000)
        .summarize();
    println!("\nFinal Stats\n{}", summary);
}
