use chrono::Utc;
use rand::SeedableRng;
use relearn::agents::BuildAgent;
use relearn::envs::{CartPole, Environment};
use relearn::logging::{DisplayLogger, TensorBoardLogger};
use relearn::simulation::{train_parallel, TrainParallelConfig};
use relearn::torch::{
    agents::ActorCriticConfig,
    critic::GaeConfig,
    modules::{AsSeq, MlpConfig},
    optimizers::{AdamConfig, ConjugateGradientOptimizerConfig},
    updaters::{CriticLossUpdateRule, TrpoPolicyUpdateRule, WithOptimizer},
};
use relearn::Prng;
use std::path::PathBuf;
use std::time::Duration;
use tch::Device;

type Module = AsSeq<MlpConfig>;

fn main() {
    let agent_config: ActorCriticConfig<
        Module,
        WithOptimizer<TrpoPolicyUpdateRule, ConjugateGradientOptimizerConfig>,
        GaeConfig<Module>,
        WithOptimizer<CriticLossUpdateRule, AdamConfig>,
    > = ActorCriticConfig {
        device: Device::Cuda(0),
        ..Default::default()
    };
    let training_config = TrainParallelConfig {
        num_periods: 10_000,
        num_threads: num_cpus::get(),
        min_workers_steps: 10_000,
    };

    let mut rng = Prng::seed_from_u64(0);
    let env = CartPole::default().with_step_limit(500);
    let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();

    let mut log_dir: PathBuf = ["data", "cartpole-trpo"].iter().collect();
    log_dir.push(Utc::now().format("%Y-%m-%d_%H-%M-%S").to_string());
    println!("Logging to {:?}", log_dir);
    let mut logger = (
        DisplayLogger::default(),
        TensorBoardLogger::new(log_dir, Duration::from_millis(200)),
    );

    train_parallel(
        &mut agent,
        &env,
        &training_config,
        &mut Prng::from_rng(&mut rng).unwrap(),
        &mut rng,
        &mut logger,
    );
}
