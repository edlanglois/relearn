use super::agent::{ForFiniteFinite, ForMetaFiniteFinite};
use super::{AgentDef, MultiThreadAgentDef};
use crate::agents::{Agent, BuildAgent, BuildManagerAgent, ManagerAgent};
use crate::envs::{
    Bandit, BuildEnv, BuildEnvError, BuildPomdp, Chain as ChainEnv, DirichletRandomMdps,
    EnvStructure, Environment, MemoryGame as MemoryGameEnv, MetaPomdp, OneHotBandits, Pomdp,
    UniformBernoulliBandits, WithStepLimit,
};
use crate::error::RLError;
use crate::logging::Loggable;
use crate::simulation::{
    hooks::StepLogger, GenericSimulationHook, MultiThreadSimulatorConfig, RunSimulation,
    SimulationHook, Simulator,
};
use crate::spaces::ElementRefInto;
use crate::utils::distributions::{Bernoulli, Bounded, Deterministic, FromMean};
use rand::distributions::Distribution;
use std::borrow::{Borrow, Cow};
use std::error::Error;
use std::marker::PhantomData;

/// Environment definition
#[derive(Debug, Clone)]
pub enum EnvDef {
    /// Multi-armed bandit with fixed arm distributions.
    Bandit(DistributionType, BanditMeanRewards),
    /// The Chain environment,
    Chain(ChainEnv),
    /// The Memory Game environment
    MemoryGame(MemoryGameEnv),
    /// Meta one-hot bandits environment
    MetaOneHotBandits(MetaPomdp<OneHotBandits>),
    /// Meta uniform bernoulli bandits environment
    MetaUniformBernoulliBandits(MetaPomdp<UniformBernoulliBandits>),
    /// Meta dirichlet MDPs environment
    MetaDirichletMdps(MetaPomdp<WithStepLimit<DirichletRandomMdps>>),
}

/// Definition of a scalar floating-point distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionType {
    Deterministic,
    Bernoulli,
}

/// Bandit mean arm reads
#[derive(Debug, Clone, PartialEq)]
pub struct BanditMeanRewards {
    pub mean_rewards: Vec<f64>,
}

impl Default for BanditMeanRewards {
    fn default() -> Self {
        Self {
            mean_rewards: vec![0.2, 0.8],
        }
    }
}

#[derive(Debug, Clone)]
struct BanditConfig<'a, D> {
    mean_rewards: Cow<'a, [f64]>,
    // `fn() -> D` is used so that BanditConfig is Send & Sync
    // Reference: https://stackoverflow.com/a/50201389/1267562
    distribution: PhantomData<fn() -> D>,
}

impl<'a, D> From<&'a BanditMeanRewards> for BanditConfig<'a, D> {
    fn from(config: &'a BanditMeanRewards) -> Self {
        Self {
            mean_rewards: Cow::from(&config.mean_rewards),
            distribution: PhantomData,
        }
    }
}

impl<D> From<BanditMeanRewards> for BanditConfig<'static, D> {
    fn from(config: BanditMeanRewards) -> Self {
        Self {
            mean_rewards: Cow::from(config.mean_rewards),
            distribution: PhantomData,
        }
    }
}

impl<'a, D: FromMean<f64>> BuildPomdp for BanditConfig<'a, D>
where
    D: FromMean<f64> + Distribution<f64> + Bounded<f64>,
    D::Error: Error + 'static,
{
    type State = <Self::Pomdp as Pomdp>::State;
    type Observation = <Self::Pomdp as Pomdp>::Observation;
    type Action = <Self::Pomdp as Pomdp>::Action;
    type ObservationSpace = <Self::Pomdp as EnvStructure>::ObservationSpace;
    type ActionSpace = <Self::Pomdp as EnvStructure>::ActionSpace;
    type Pomdp = Bandit<D>;

    fn build_pomdp(&self) -> Result<Self::Pomdp, BuildEnvError> {
        let mean_rewards: &[f64] = self.mean_rewards.borrow();
        Bandit::from_means(mean_rewards).map_err(|e| BuildEnvError::Boxed(Box::new(e)))
    }
}

/// Construct a boxed agent-environment simulation
fn boxed_simulation<EC, AC, H>(
    env_def: &EC,
    agent_def: &AC,
    env_seed: u64,
    agent_seed: u64,
    hook: H,
) -> Result<Box<dyn RunSimulation>, RLError>
where
    EC: BuildEnv + ?Sized,
    EC::ObservationSpace: ElementRefInto<Loggable> + 'static,
    EC::Observation: Clone + 'static,
    EC::ActionSpace: ElementRefInto<Loggable> + 'static,
    EC::Action: 'static,
    EC::Environment: 'static,
    AC: BuildAgent<
            dyn EnvStructure<
                ObservationSpace = EC::ObservationSpace,
                ActionSpace = EC::ActionSpace,
            >,
        > + ?Sized,
    AC::Agent: 'static,
    H: SimulationHook<EC::Observation, EC::Action> + 'static,
{
    // Boxed so that we we avoid creating a copy of the simulator code for each environment type
    let env = Box::new(env_def.build_env(env_seed)?);

    // Not boxed because this is expected to be called with (wrapped) AgentDef,
    // which already boxes the output agent.
    let agent = agent_def.build_agent(&env, agent_seed)?;

    let log_hook = StepLogger::new(env.observation_space(), env.action_space());

    // Reduce to an environment trait object
    let env: Box<dyn Environment<Action = _, Observation = _>> = env;
    Ok(Box::new(Simulator::new(env, agent, (log_hook, hook))))
}

/// Construct a boxed parallel agent-environment simulation
fn boxed_parallel_simulation<EC, AC, H>(
    sim_config: &MultiThreadSimulatorConfig,
    env_def: &EC,
    agent_def: &AC,
    env_seed: u64,
    agent_seed: u64,
    hook: H,
) -> Result<Box<dyn RunSimulation>, RLError>
where
    EC: BuildEnv + Clone + Send + Sync + ?Sized + 'static,
    EC::ObservationSpace: ElementRefInto<Loggable> + Clone + Send + 'static,
    EC::Observation: Clone + 'static,
    EC::ActionSpace: ElementRefInto<Loggable> + Clone + Send + 'static,
    EC::Action: 'static,
    EC::Environment: 'static,
    AC: BuildManagerAgent<
            dyn EnvStructure<
                ObservationSpace = EC::ObservationSpace,
                ActionSpace = EC::ActionSpace,
            >,
        > + ?Sized,
    AC::ManagerAgent: 'static,
    <AC::ManagerAgent as ManagerAgent>::Worker: Agent<EC::Observation, EC::Action> + 'static,
    H: SimulationHook<EC::Observation, EC::Action> + Clone + Send + 'static,
{
    let env = env_def.build_env(env_seed)?;

    // Not boxed because this is expected to be called with (wrapped) AgentManagerDef,
    // which already boxes the output agent.
    let agent = agent_def.build_manager_agent(&env, agent_seed)?;

    let log_hook = StepLogger::new(env.observation_space(), env.action_space());

    Ok(sim_config.build_simulator(env_def.clone(), agent, (log_hook, hook)))
}

impl EnvDef {
    /// Construct a boxed simulation for this environment and a given agent definition.
    ///
    /// # Args
    /// * `agent_def` - Agent definition.
    /// * `env_seed` - Random seed used for the environment.
    /// * `agent_seed` - Random seed used for the agent.
    /// * `hook` - A hook called on each step of the simulation. Pass () for no hook.
    pub fn build_simulation<H>(
        &self,
        agent_def: &AgentDef,
        env_seed: u64,
        agent_seed: u64,
        hook: H,
    ) -> Result<Box<dyn RunSimulation>, RLError>
    where
        H: GenericSimulationHook + 'static,
    {
        macro_rules! make_simulation {
            ($env_config:expr, $agent_builder:ty) => {
                boxed_simulation(
                    $env_config,
                    &<$agent_builder>::new(agent_def),
                    env_seed,
                    agent_seed,
                    hook,
                )
            };
        }

        use EnvDef::*;
        match self {
            Bandit(dist_type, means) => match dist_type {
                DistributionType::Deterministic => {
                    let config = BanditConfig::<Deterministic<f64>>::from(means);
                    make_simulation!(&config, ForFiniteFinite<_>)
                }
                DistributionType::Bernoulli => {
                    let config = BanditConfig::<Bernoulli>::from(means);
                    make_simulation!(&config, ForFiniteFinite<_>)
                }
            },
            Chain(config) => {
                make_simulation!(config, ForFiniteFinite<_>)
            }
            MemoryGame(config) => {
                make_simulation!(config, ForFiniteFinite<_>)
            }
            MetaOneHotBandits(config) => {
                make_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaUniformBernoulliBandits(config) => {
                make_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaDirichletMdps(config) => {
                make_simulation!(config, ForMetaFiniteFinite<_>)
            }
        }
    }

    // TODO: De-deuplicate with build_simulatior
    /// Construct a multi-thread boxed simulation for this environment and given agent definition.
    ///
    /// # Args
    /// * `sim_config` - Simulator configuration.
    /// * `agent_def` - Agent definition.
    /// * `env_seed` - Random seed used for the environment.
    /// * `agent_seed` - Random seed used for the agent.
    /// * `hook` - A hook called on each step of the simulation. Pass () for no hook.
    pub fn build_parallel_simulation<H>(
        &self,
        sim_config: &MultiThreadSimulatorConfig,
        agent_def: &MultiThreadAgentDef,
        env_seed: u64,
        agent_seed: u64,
        hook: H,
    ) -> Result<Box<dyn RunSimulation>, RLError>
    where
        H: GenericSimulationHook + Clone + Send + 'static,
    {
        /// Construct a boxed agent-environment simulation
        macro_rules! make_simulation {
            ($env_config:expr, $agent_builder:ty) => {{
                boxed_parallel_simulation(
                    sim_config,
                    $env_config,
                    &<$agent_builder>::new(agent_def),
                    env_seed,
                    agent_seed,
                    hook,
                )
            }};
        }

        use EnvDef::*;
        match self {
            Bandit(dist_type, means) => match dist_type {
                // TODO: Avoid double clone (in config assignment and in boxed_simulation)
                DistributionType::Deterministic => {
                    let config = BanditConfig::<Deterministic<f64>>::from(means.clone());
                    make_simulation!(&config, ForFiniteFinite<_>)
                }
                DistributionType::Bernoulli => {
                    let config = BanditConfig::<Bernoulli>::from(means.clone());
                    make_simulation!(&config, ForFiniteFinite<_>)
                }
            },
            Chain(config) => {
                make_simulation!(config, ForFiniteFinite<_>)
            }
            MemoryGame(config) => {
                make_simulation!(config, ForFiniteFinite<_>)
            }
            MetaOneHotBandits(config) => {
                make_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaUniformBernoulliBandits(config) => {
                make_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaDirichletMdps(config) => {
                make_simulation!(config, ForMetaFiniteFinite<_>)
            }
        }
    }
}
