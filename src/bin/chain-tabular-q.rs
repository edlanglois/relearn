use relearn::agents::{
    buffers::SerialBufferConfig, BatchedUpdatesConfig, BuildBatchAgent, TabularQLearningAgentConfig,
};
use relearn::envs::{BuildEnv, Chain};
use relearn::logging::CLILoggerConfig;
use relearn::simulation::train_parallel;

fn main() {
    let env_config = Chain::default();
    let history_buffer_config = SerialBufferConfig {
        soft_threshold: 1_000_000,
        hard_threshold: 1_100_000,
    };
    let agent_config = BatchedUpdatesConfig {
        agent_config: TabularQLearningAgentConfig::default(),
        history_buffer_config,
    };

    let logger_config = CLILoggerConfig::default();

    let mut agent = agent_config
        .build_batch_agent(&env_config.build_env(0).unwrap(), 0)
        .unwrap();
    train_parallel(
        &mut agent,
        &env_config,
        10,
        num_cpus::get(),
        1,
        &mut logger_config.build_logger(),
    )
    .unwrap();
}
