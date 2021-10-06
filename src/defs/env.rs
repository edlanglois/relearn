use super::agent::{ForFiniteFinite, ForMetaFiniteFinite};
use super::{AgentDef, HooksDef, MultiThreadAgentDef};
use crate::agents::{Agent, BuildAgent, BuildManagerAgent, ManagerAgent};
use crate::envs::{
    Bandit, BuildEnv, BuildEnvError, BuildPomdp, Chain as ChainEnv, DirichletRandomMdps,
    EnvStructure, MemoryGame as MemoryGameEnv, MetaPomdp, OneHotBandits, Pomdp,
    UniformBernoulliBandits, WithStepLimit,
};
use crate::logging::Loggable;
use crate::simulation::{
    BuildSimulationHook, ParallelSimulatorConfig, SerialSimulator, Simulator, SimulatorError,
};
use crate::spaces::ElementRefInto;
use crate::utils::distributions::{Bernoulli, Bounded, Deterministic, FromMean};
use rand::distributions::Distribution;
use std::borrow::Borrow;
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
struct BanditConfig<D> {
    mean_rewards: Vec<f64>,
    // `fn() -> D` is used so that BanditConfig is Send & Sync
    // Reference: https://stackoverflow.com/a/50201389/1267562
    distribution: PhantomData<fn() -> D>,
}

impl<D> From<&BanditMeanRewards> for BanditConfig<D> {
    fn from(config: &BanditMeanRewards) -> Self {
        Self {
            mean_rewards: config.mean_rewards.clone(),
            distribution: PhantomData,
        }
    }
}

impl<D: FromMean<f64>> BuildPomdp for BanditConfig<D>
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
fn boxed_simulation<EC, AC, HC>(
    env_def: &EC,
    agent_def: &AC,
    hook_def: &HC,
) -> Result<Box<dyn Simulator>, SimulatorError>
where
    EC: BuildEnv + Clone + 'static,
    EC::ObservationSpace: ElementRefInto<Loggable> + Clone + 'static, // TODO: Remove Clone
    EC::Observation: Clone + 'static,
    EC::ActionSpace: ElementRefInto<Loggable> + Clone + 'static, // TODO: Remove Clone when using BuildHook
    EC::Action: 'static,
    EC::Environment: 'static,
    AC: BuildAgent<EC::ObservationSpace, EC::ActionSpace> + Clone + 'static,
    AC::Agent: 'static,
    HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace> + Clone + 'static,
    HC::Hook: Send + 'static,
{
    Ok(Box::new(SerialSimulator::new(
        env_def.clone(),
        agent_def.clone(),
        hook_def.clone(),
    )))
}

/// Construct a boxed parallel agent-environment simulation
fn boxed_parallel_simulation<EC, AC, HC>(
    sim_config: &ParallelSimulatorConfig,
    env_def: &EC,
    agent_def: &AC,
    hook_def: &HC,
    env_seed: u64,
    agent_seed: u64,
) -> Result<Box<dyn Simulator>, SimulatorError>
where
    EC: BuildEnv + Clone + 'static,
    EC::ObservationSpace: ElementRefInto<Loggable> + Clone + Send + 'static,
    EC::Observation: Clone + 'static,
    EC::ActionSpace: ElementRefInto<Loggable> + Clone + Send + 'static,
    EC::Action: 'static,
    EC::Environment: Send + 'static,
    AC: BuildManagerAgent<EC::ObservationSpace, EC::ActionSpace> + ?Sized,
    AC::ManagerAgent: 'static,
    <AC::ManagerAgent as ManagerAgent>::Worker: Agent<EC::Observation, EC::Action> + 'static,
    HC: BuildSimulationHook<EC::ObservationSpace, EC::ActionSpace> + Clone + 'static,
    HC::Hook: Send + 'static,
{
    let env = env_def.build_env(env_seed)?;

    // Not boxed because this is expected to be called with (wrapped) AgentManagerDef,
    // which already boxes the output agent.
    let agent = agent_def.build_manager_agent(&env, agent_seed)?;

    Ok(sim_config.build_simulator(env_def.clone(), agent, hook_def.clone()))
}

impl EnvDef {
    /// Construct a boxed simulation for this environment and a given agent definition.
    ///
    /// # Args
    /// * `agent_def` - Agent definition.
    /// * `env_seed` - Random seed used for the environment.
    /// * `agent_seed` - Random seed used for the agent.
    /// * `hook` - A hook called on each step of the simulation. Pass () for no hook.
    pub fn build_simulation(
        &self,
        agent_def: &AgentDef,
        hook_def: &HooksDef,
    ) -> Result<Box<dyn Simulator>, SimulatorError> {
        macro_rules! make_simulation {
            ($env_config:expr, $agent_builder:ty) => {
                boxed_simulation(
                    $env_config,
                    &<$agent_builder>::new(agent_def.clone()), // TODO: Avoid clone
                    hook_def,
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
    pub fn build_parallel_simulation(
        &self,
        sim_config: &ParallelSimulatorConfig,
        agent_def: &MultiThreadAgentDef,
        hook_def: &HooksDef,
        env_seed: u64,
        agent_seed: u64,
    ) -> Result<Box<dyn Simulator>, SimulatorError> {
        /// Construct a boxed agent-environment simulation
        macro_rules! make_simulation {
            ($env_config:expr, $agent_builder:ty) => {{
                boxed_parallel_simulation(
                    sim_config,
                    $env_config,
                    &<$agent_builder>::new(agent_def),
                    hook_def,
                    env_seed,
                    agent_seed,
                )
            }};
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
}
