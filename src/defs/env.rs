use super::agent::{ForFiniteFinite, ForMetaFiniteFinite};
use super::{AgentDef, MultiThreadAgentDef};
use crate::agents::{BuildAgent, BuildManagerAgent};
use crate::envs::{
    Bandit, BuildEnv, BuildEnvError, BuildPomdp, Chain as ChainEnv, DirichletRandomMdps,
    EnvStructure, MemoryGame as MemoryGameEnv, MetaPomdp, OneHotBandits, Pomdp,
    UniformBernoulliBandits, WithStepLimit,
};
use crate::error::RLError;
use crate::simulation::{
    hooks::StepLogger, GenericSimulationHook, MultiThreadSimulatorConfig, RunSimulation, Simulator,
};
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
        /// Construct a boxed agent-environment simulation
        macro_rules! boxed_simulation {
            ($env_config:expr, $agent_builder:ty) => {{
                let env = Box::new($env_config.build_env(env_seed)?);
                let agent = <$agent_builder>::new(agent_def).build_agent(&env, agent_seed)?;
                let log_hook = StepLogger::new(env.observation_space(), env.action_space());
                Box::new(Simulator::new(env, agent, (log_hook, hook)))
            }};
        }

        use EnvDef::*;
        Ok(match self {
            Bandit(dist_type, means) => match dist_type {
                DistributionType::Deterministic => {
                    let config = BanditConfig::<Deterministic<f64>>::from(means);
                    boxed_simulation!(config, ForFiniteFinite<_>)
                }
                DistributionType::Bernoulli => {
                    let config = BanditConfig::<Bernoulli>::from(means);
                    boxed_simulation!(config, ForFiniteFinite<_>)
                }
            },
            Chain(config) => {
                boxed_simulation!(config, ForFiniteFinite<_>)
            }
            MemoryGame(config) => {
                boxed_simulation!(config, ForFiniteFinite<_>)
            }
            MetaOneHotBandits(config) => {
                boxed_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaUniformBernoulliBandits(config) => {
                boxed_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaDirichletMdps(config) => {
                boxed_simulation!(config, ForMetaFiniteFinite<_>)
            }
        })
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
        macro_rules! boxed_simulation {
            ($env_config:expr, $agent_builder:ty) => {{
                let env = $env_config.build_env(env_seed)?;
                let agent =
                    <$agent_builder>::new(agent_def).build_manager_agent(&env, agent_seed)?;
                let log_hook = StepLogger::new(env.observation_space(), env.action_space());
                sim_config.build_simulator($env_config.clone(), agent, (log_hook, hook))
            }};
        }

        use EnvDef::*;
        Ok(match self {
            Bandit(dist_type, means) => match dist_type {
                // TODO: Avoid double clone (in config assignment and in boxed_simulation)
                DistributionType::Deterministic => {
                    let config = BanditConfig::<Deterministic<f64>>::from(means.clone());
                    boxed_simulation!(config, ForFiniteFinite<_>)
                }
                DistributionType::Bernoulli => {
                    let config = BanditConfig::<Bernoulli>::from(means.clone());
                    boxed_simulation!(config, ForFiniteFinite<_>)
                }
            },
            Chain(config) => {
                boxed_simulation!(config, ForFiniteFinite<_>)
            }
            MemoryGame(config) => {
                boxed_simulation!(config, ForFiniteFinite<_>)
            }
            MetaOneHotBandits(config) => {
                boxed_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaUniformBernoulliBandits(config) => {
                boxed_simulation!(config, ForMetaFiniteFinite<_>)
            }
            MetaDirichletMdps(config) => {
                boxed_simulation!(config, ForMetaFiniteFinite<_>)
            }
        })
    }
}
