/// Cart-Pole DQN Example
use rand::SeedableRng;
use relearn::agents::{ActorMode, Agent, BuildAgent};
use relearn::envs::{
    CartPole, EnvStructure, Environment, VisibleStepLimit, WithVisibleStepLimit, Wrap,
};
use relearn::logging::{ByCounter, ByTime, DisplayLogger, TensorBoardLogger};
use relearn::simulation::{train_parallel, SimSeed, StepsIter, TrainParallelConfig};
use relearn::torch::agents::schedules::DataCollectionSchedule;
use relearn::torch::agents::DqnConfig;
use relearn::torch::modules::MlpConfig;
use relearn::torch::optimizers::AdamConfig;
use relearn::Prng;
use std::env;
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::Duration;
use tch::Device;

type AgentConfig = DqnConfig<MlpConfig, AdamConfig>;

fn main() {
    let max_episode_len = 500;
    let env = CartPole::default().wrap(VisibleStepLimit::new(max_episode_len));
    println!("Env:\n{:#?}\n", env);

    let args: Vec<String> = env::args().collect();
    match &args[1..] {
        [] => {
            let mut output_dir: PathBuf = ["data", "cartpole-dqn"].iter().collect();
            output_dir.push(chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string());
            fs::create_dir_all(&output_dir).unwrap();

            let agent_config: AgentConfig = DqnConfig {
                buffer_capacity: 50_000_000,
                update_size: DataCollectionSchedule::FirstRest {
                    first: 5_000_000,
                    rest: 100_000,
                },
                device: Device::cuda_if_available(),
                ..Default::default()
            };
            println!("Agent Config\n{:#?}\n", agent_config);
            let agent_config_path = output_dir.join("agent_config.json");
            println!("Saving agent config to {:?}", agent_config_path);
            serde_json::to_writer(File::create(agent_config_path).unwrap(), &agent_config).unwrap();

            let training_config = TrainParallelConfig {
                num_periods: 500,
                num_threads: num_cpus::get(),
                min_worker_steps: max_episode_len as usize,
            };
            println!("Training Config\n{:#?}\n", training_config);

            let mut rng = Prng::seed_from_u64(0);
            let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();

            println!("Logging to {:?}", output_dir);
            let log_on_name = ["agent_update", "count"];
            let mut logger = (
                DisplayLogger::new(ByTime::new(Duration::from_secs(1))),
                TensorBoardLogger::new(ByCounter::of_path(log_on_name, 1), &output_dir),
            );

            train_parallel(
                &mut agent,
                &env,
                &training_config,
                &mut Prng::from_rng(&mut rng).unwrap(),
                &mut rng,
                &mut logger,
            );
            drop(logger); // Flush output before the following prints

            let actor_path = output_dir.join("actor.cbor");
            println!("Saving actor to {:?}", actor_path);
            serde_cbor::to_writer(
                File::create(&actor_path).unwrap(),
                &agent.actor(ActorMode::Evaluation),
            )
            .unwrap();
            println!("To evaluate the actor run\n{:?} {:?}", args[0], actor_path);
        }
        [actor_path] => {
            println!("Loading actor from {:?}", actor_path);
            #[allow(clippy::type_complexity)]
            let actor: <<AgentConfig as BuildAgent<
                <WithVisibleStepLimit<CartPole> as EnvStructure>::ObservationSpace,
                <WithVisibleStepLimit<CartPole> as EnvStructure>::ActionSpace,
                <WithVisibleStepLimit<CartPole> as EnvStructure>::FeedbackSpace,
            >>::Agent as Agent<_, _>>::Actor =
                serde_cbor::from_reader(File::open(&actor_path).unwrap()).unwrap();

            let summary = env
                .run(&actor, SimSeed::Root(0), ())
                .take(10_000)
                .summarize();
            println!("\nEvaluation Stats\n{:.3}", summary);
        }
        _ => eprintln!("Usage: {} [saved_agent_file]", args[0]),
    }
}
