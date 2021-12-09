use relearn::agents::{
    buffers::SimpleBufferConfig, ActorMode, BatchedUpdatesConfig, BuildBatchAgent, PureAsActor,
    SetActorMode, TabularQLearningAgentConfig,
};
use relearn::envs::{BuildEnv, Chain, Environment};
use relearn::logging::CLILoggerConfig;
use relearn::simulation::{train_parallel, SimulationSummary};

fn main() {
    let env_config = Chain::default();
    let history_buffer_config = SimpleBufferConfig::with_threshold(1_000_000);
    let agent_config = BatchedUpdatesConfig {
        agent_config: TabularQLearningAgentConfig::default(),
        history_buffer_config,
    };

    let logger_config = CLILoggerConfig::default();

    let env = env_config.build_env(0).unwrap();
    let mut agent = agent_config.build_batch_agent(&env, 0).unwrap();

    {
        // TODO: Debug set_actor_mode agent.make_actor()
        let mut actor = PureAsActor::new(agent.clone(), 0);
        actor.set_actor_mode(ActorMode::Release);
        let summary = SimulationSummary::from_steps(env.run(actor, ()).take(10_000));
        println!("Initial Stats\n{}", summary);
    }

    train_parallel(
        &mut agent,
        &env_config,
        10,
        num_cpus::get(),
        1,
        &mut logger_config.build_logger(),
    )
    .unwrap();

    let mut agent = PureAsActor::new(agent, 0);
    let env = env_config.build_env(1).unwrap();
    agent.set_actor_mode(ActorMode::Release);
    let summary = SimulationSummary::from_steps(env.run(agent, ()).take(10_000));
    println!("\nFinal Stats\n{}", summary);
}
