//! Benchmark `Actor::act` for various agents.
use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, Criterion,
};
use rand::SeedableRng;
use relearn::agents::{
    Actor, ActorMode, Agent, BetaThompsonSamplingAgentConfig, BuildAgent, RandomAgentConfig,
    TabularQLearningAgentConfig, UCB1AgentConfig,
};
use relearn::envs::{EnvStructure, Environment, Successor};
use relearn::feedback::Reward;
use relearn::logging::StatsLogger;
use relearn::spaces::{BooleanSpace, IndexSpace, IntervalSpace};
use relearn::torch::{
    agents::{critics::RewardToGoConfig, policies::ReinforceConfig, ActorCriticConfig},
    modules::{GruConfig, MlpConfig},
};
use relearn::Prng;

const RING_ENV_SIZE: usize = 5;

/// Trivial non-episodic environment consisting of a ring of integers `0..RING_ENV_SIZE`.
///
/// The agent can move forward or backward along the ring, with wrapping.
/// The reward is always 0.
pub struct RingEnv;

impl Environment for RingEnv {
    type State = usize;
    type Observation = usize;
    type Action = bool;
    type Feedback = Reward;

    fn initial_state(&self, _: &mut Prng) -> Self::State {
        0
    }
    fn observe(&self, state: &Self::State, _: &mut Prng) -> Self::Observation {
        *state
    }
    fn step(
        &self,
        mut state: Self::State,
        action: &Self::Action,
        _: &mut Prng,
        _: &mut dyn StatsLogger,
    ) -> (Successor<Self::State>, Self::Feedback) {
        let new_state = match *action {
            false => state.checked_sub(1).unwrap_or(RING_ENV_SIZE - 1),
            true => {
                state += 1;
                if state >= RING_ENV_SIZE {
                    state = 0
                }
                state
            }
        };
        (Successor::Continue(new_state), Reward(0.0))
    }
}

impl EnvStructure for RingEnv {
    type ObservationSpace = IndexSpace;
    type ActionSpace = BooleanSpace;
    type FeedbackSpace = IntervalSpace<Reward>;

    fn observation_space(&self) -> Self::ObservationSpace {
        IndexSpace::new(RING_ENV_SIZE)
    }
    fn action_space(&self) -> Self::ActionSpace {
        BooleanSpace
    }
    fn feedback_space(&self) -> Self::FeedbackSpace {
        IntervalSpace::new(Reward(0.0), Reward(0.0))
    }
    fn discount_factor(&self) -> f64 {
        0.99
    }
}

/// Benchmark `Actor::act` of an agent's evaluation actor on a trivial non-episodic environment.
fn benchmark_agent_act<M, TC>(group: &mut BenchmarkGroup<M>, name: &str, agent_config: &TC)
where
    M: Measurement,
    TC: BuildAgent<IndexSpace, BooleanSpace, IntervalSpace<Reward>>,
{
    let mut rng = Prng::seed_from_u64(0);

    let env = RingEnv;
    let agent = agent_config.build_agent(&env, &mut rng).unwrap();
    let actor = agent.actor(ActorMode::Evaluation);

    let mut env_state = env.initial_state(&mut rng);
    let mut obs = env.observe(&env_state, &mut rng);
    let mut actor_state = actor.initial_state(&mut rng);
    group.bench_function(name, |b| {
        b.iter(|| {
            // This is the thing we want to benchmark
            let action = actor.act(&mut actor_state, &obs, &mut rng);
            // Updating the environment state to provide the agent with different inputs.
            // Ideally this would be excluded from the benchmark, but at least it should be fast.
            // RingEnv episodes never terminate so don't have to worry about episode transitions.
            env_state = env
                .step(env_state, &action, &mut rng, &mut ())
                .0
                .into_continue()
                .unwrap();
            obs = env.observe(&env_state, &mut rng);
        })
    });
}

fn bench_agents_act(c: &mut Criterion) {
    let mut group = c.benchmark_group("agents_act");
    benchmark_agent_act(&mut group, "random", &RandomAgentConfig);
    benchmark_agent_act(
        &mut group,
        "beta_thompson_sampling",
        &BetaThompsonSamplingAgentConfig::default(),
    );
    benchmark_agent_act(&mut group, "ucb1", &UCB1AgentConfig::default());
    benchmark_agent_act(
        &mut group,
        "tabular_q_learning",
        &TabularQLearningAgentConfig::default(),
    );
    benchmark_agent_act(
        &mut group,
        "actor_critic_mlp",
        &ActorCriticConfig::<ReinforceConfig<MlpConfig>, RewardToGoConfig>::default(),
    );
    benchmark_agent_act(
        &mut group,
        "actor_critic_gru",
        &ActorCriticConfig::<ReinforceConfig<GruConfig>, RewardToGoConfig>::default(),
    );
}

criterion_group!(benches, bench_agents_act);
criterion_main!(benches);
