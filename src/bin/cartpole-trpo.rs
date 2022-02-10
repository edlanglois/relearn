use rand::SeedableRng;
use relearn::agents::BuildAgent;
use relearn::envs::{CartPole, Environment};
use relearn::logging::DisplayLogger;
use relearn::simulation::{train_parallel, TrainParallelConfig};
use relearn::torch::{
    agents::ActorCriticConfig,
    critic::GaeConfig,
    modules::{AsSeq, MlpConfig},
    optimizers::{AdamConfig, ConjugateGradientOptimizerConfig},
    updaters::{CriticLossUpdateRule, TrpoPolicyUpdateRule, WithOptimizer},
};
use relearn::Prng;
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
    let mut logger = DisplayLogger::default();

    train_parallel(
        &mut agent,
        &env,
        &training_config,
        &mut Prng::from_rng(&mut rng).unwrap(),
        &mut rng,
        &mut logger,
    );
}
