use rand::{Rng, SeedableRng};
use relearn::agents::{ActorMode, Agent, BuildAgent};
use relearn::envs::{BuildEnv, Environment, FirstPlayerView, FruitGame, StepLimit, WithStepLimit};
use relearn::logging::DisplayLogger;
use relearn::simulation::{train_parallel, SimulationSummary, TrainParallelConfig};
use relearn::torch::{
    agents::ActorCriticConfig,
    critic::GaeConfig,
    modules::GruMlpConfig,
    optimizers::AdamConfig,
    updaters::{CriticLossUpdateRule, PpoPolicyUpdateRule, WithOptimizer},
};
use relearn::Prng;
use tch::Device;

fn main() {
    let env_config = WithStepLimit::new(
        FirstPlayerView::new(FruitGame::<5, 5, 5, 5>::default()),
        StepLimit::new(50),
    );

    let agent_config: ActorCriticConfig<
        GruMlpConfig,
        WithOptimizer<PpoPolicyUpdateRule, AdamConfig>,
        GaeConfig<GruMlpConfig>,
        WithOptimizer<CriticLossUpdateRule, AdamConfig>,
    > = ActorCriticConfig {
        device: Device::Cuda(0),
        ..Default::default()
    };
    let training_config = TrainParallelConfig {
        num_periods: 200,
        num_threads: num_cpus::get(),
        min_workers_steps: 10,
    };
    let mut rng = Prng::seed_from_u64(0);
    let env = env_config.build_env(&mut rng).unwrap();
    let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();
    let mut logger = DisplayLogger::default();

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
