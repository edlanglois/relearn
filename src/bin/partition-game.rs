use rand::SeedableRng;
use relearn::agents::{ActorMode, Agent, BuildAgent};
use relearn::envs::{BuildEnv, Environment, PartitionGame};
use relearn::logging::DisplayLogger;
use relearn::simulation::{train_parallel, SimSeed, SimulationSummary, TrainParallelConfig};
use relearn::torch::{
    agents::ActorCriticConfig,
    critic::GaeConfig,
    modules::GruMlpConfig,
    optimizers::{AdamConfig, ConjugateGradientOptimizerConfig},
    updaters::{CriticLossUpdateRule, TrpoPolicyUpdateRule, WithOptimizer},
};
use relearn::Prng;
use tch::Device;

fn main() {
    let env_config = PartitionGame::default().with_step_limit(1000);

    let agent_config: ActorCriticConfig<
        GruMlpConfig,
        WithOptimizer<TrpoPolicyUpdateRule, ConjugateGradientOptimizerConfig>,
        // WithOptimizer<PpoPolicyUpdateRule, AdamConfig>,
        GaeConfig<GruMlpConfig>,
        WithOptimizer<CriticLossUpdateRule, AdamConfig>,
    > = ActorCriticConfig {
        device: Device::Cuda(0),
        ..Default::default()
    };
    let training_config = TrainParallelConfig {
        num_periods: 1000,
        num_threads: num_cpus::get(),
        min_workers_steps: 5_000,
    };
    let mut rng = Prng::seed_from_u64(0);
    let env = env_config.build_env(&mut rng).unwrap();
    let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();
    let mut logger = DisplayLogger::default();

    {
        let summary = SimulationSummary::from_steps(
            env.run(&agent.actor(ActorMode::Evaluation), SimSeed::Root(0), ())
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
        env.run(&agent.actor(ActorMode::Evaluation), SimSeed::Root(0), ())
            .take(10_000),
    );
    println!("\nFinal Stats\n{}", summary);
}
