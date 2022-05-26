use rand::SeedableRng;
use relearn::agents::buffers::{HistoryDataBound, NullBuffer};
use relearn::agents::{
    Actor, ActorMode, Agent, AgentPair, BatchUpdate, BuildAgent, BuildAgentError,
};
use relearn::envs::{
    fruit, BuildEnv, EnvStructure, Environment, FruitGame, LatentStepLimit, WithLatentStepLimit,
};
use relearn::logging::{DisplayLogger, StatsLogger};
use relearn::simulation::{train_parallel, SimSeed, StepsIter, TrainParallelConfig};
use relearn::spaces::{IndexedTypeSpace, Space};
use relearn::torch::{
    agents::{
        critic::GaeConfig, learning_critic::GradOptConfig, learning_policy::PpoConfig,
        ActorCriticConfig,
    },
    modules::{ChainConfig, GruConfig, MlpConfig},
};
use relearn::Prng;
use std::cmp::Ordering;
use tch::Device;

/// Lazy expert player for [`FruitGame`]. Picks up one of the correct kind of fruit then stops.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FruitLazyExpert;

impl<const W: usize, const H: usize>
    BuildAgent<fruit::PrincipalObsSpace<W, H>, IndexedTypeSpace<fruit::Move>> for FruitLazyExpert
{
    type Agent = Self;

    fn build_agent(
        &self,
        _: &dyn EnvStructure<
            ObservationSpace = fruit::PrincipalObsSpace<W, H>,
            ActionSpace = IndexedTypeSpace<fruit::Move>,
        >,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(*self)
    }
}

impl<const W: usize, const H: usize> Agent<fruit::PrincipalObs<W, H>, fruit::Move>
    for FruitLazyExpert
{
    type Actor = Self;

    fn actor(&self, _: ActorMode) -> Self::Actor {
        *self
    }
}

pub const fn abs_diff(a: usize, b: usize) -> usize {
    if a > b {
        a - b
    } else {
        b - a
    }
}

impl<const W: usize, const H: usize> Actor<fruit::PrincipalObs<W, H>, fruit::Move>
    for FruitLazyExpert
{
    /// Whether a fruit has been collected.
    type EpisodeState = bool;

    fn initial_state(&self, _: &mut Prng) -> Self::EpisodeState {
        false
    }

    fn act(
        &self,
        fruit_collected: &mut Self::EpisodeState,
        obs: &fruit::PrincipalObs<W, H>,
        _: &mut Prng,
    ) -> fruit::Move {
        if *fruit_collected {
            return fruit::Move::Still;
        }
        let mid_i = H / 2;
        let mid_j = W / 2;
        let mut fruit_pos = None;
        let mut fruit_distance = H + W;
        for (i, row) in obs.visible_grid.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                if (obs.goal_is_apple && *cell == fruit::CellView::Apple)
                    || (!obs.goal_is_apple && *cell == fruit::CellView::Cherry)
                {
                    let distance = abs_diff(mid_i, i) + abs_diff(mid_j, j);
                    if distance < fruit_distance {
                        fruit_pos = Some((i, j));
                        fruit_distance = distance;
                    }
                }
            }
        }
        if fruit_distance == 1 {
            *fruit_collected = true;
        }

        let (i, j) = fruit_pos.expect("no fruit in sight");
        match (i.cmp(&mid_i), j.cmp(&mid_j)) {
            (Ordering::Less, _) => fruit::Move::Up,
            (Ordering::Greater, _) => fruit::Move::Down,
            (_, Ordering::Less) => fruit::Move::Left,
            (_, Ordering::Greater) => fruit::Move::Right,
            _ => panic!("should not be possible to be on top of fruit"),
        }
    }
}

impl<O, A> BatchUpdate<O, A> for FruitLazyExpert {
    type HistoryBuffer = NullBuffer;
    fn buffer(&self) -> Self::HistoryBuffer {
        NullBuffer
    }
    fn min_update_size(&self) -> HistoryDataBound {
        HistoryDataBound::empty()
    }
    fn batch_update<'a, I>(&mut self, _: I, _: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
    }
}

type ModelConfig = ChainConfig<MlpConfig, ChainConfig<GruConfig, MlpConfig>>;

fn main() {
    let env_config =
        WithLatentStepLimit::new(FruitGame::<5, 5, 5, 5>::default(), LatentStepLimit::new(50));

    let assistant_config: ActorCriticConfig<
        PpoConfig<ModelConfig>,
        GradOptConfig<GaeConfig<ModelConfig>>,
    > = ActorCriticConfig {
        device: Device::Cuda(0),
        ..Default::default()
    };
    let principal_config = FruitLazyExpert;
    let agent_config = AgentPair(principal_config, assistant_config);

    let training_config = TrainParallelConfig {
        num_periods: 200,
        num_threads: num_cpus::get(),
        min_worker_steps: 10_000,
    };

    let mut rng = Prng::seed_from_u64(0);
    let env = env_config.build_env(&mut rng).unwrap();
    let mut agent = agent_config.build_agent(&env, &mut rng).unwrap();
    let mut logger: DisplayLogger = DisplayLogger::default();

    {
        let summary = env
            .run(
                Agent::<<fruit::JointObsSpace<5, 5> as Space>::Element, _>::actor(
                    &agent,
                    ActorMode::Evaluation,
                ),
                SimSeed::Root(0),
                (),
            )
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
        .run(
            Agent::<<fruit::JointObsSpace<5, 5> as Space>::Element, _>::actor(
                &agent,
                ActorMode::Evaluation,
            ),
            SimSeed::Root(0),
            (),
        )
        .take(10_000)
        .summarize();
    println!("\nFinal Stats\n{}", summary);
}
