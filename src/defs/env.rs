use super::agent::{ForFiniteFinite, ForMetaFiniteFinite};
use super::{AgentDef, MultiThreadAgentDef};
use crate::agents::{Agent, BuildAgent, ManagerAgent};
use crate::envs::{
    Bandit, BuildEnv, Chain as ChainEnv, DirichletRandomMdps, EnvStructure, FixedMeansBanditConfig,
    MemoryGame as MemoryGameEnv, MetaEnvConfig, OneHotBandits, PomdpEnv, PriorMeansBanditConfig,
    StatefulMetaEnv, StepLimit, UniformBernoulliBandits, Wrapped,
};
use crate::error::RLError;
use crate::simulation::{
    hooks::StepLogger, GenericSimulationHook, MultiThreadSimulatorConfig, RunSimulation, Simulator,
};
use crate::utils::distributions::{Bernoulli, Deterministic};
use rand::distributions::Standard;

/// Environment definition
#[derive(Debug, Clone)]
pub enum EnvDef {
    /// Multi-armed bandit with fixed arm means.
    FixedMeanBandit(DistributionType, FixedMeansBanditConfig),
    /// Multi-armed bandit with uniform random arm means (sampled once on creation).
    UniformMeanBandit(DistributionType, PriorMeansBanditConfig<Standard>),
    /// The Chain environment,
    Chain(ChainEnv),
    /// The Memory Game environment
    MemoryGame(MemoryGameEnv),
    /// Meta needle-haystack bandits environment
    MetaOneHotBandits(MetaEnvConfig<OneHotBandits>),
    /// Meta uniform bernoulli bandits environment
    MetaUniformBernoulliBandits(MetaEnvConfig<UniformBernoulliBandits>),
    /// Meta dirichlet MDPs environment
    MetaDirichletMdps(MetaEnvConfig<Wrapped<DirichletRandomMdps, StepLimit>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionType {
    Deterministic,
    Bernoulli,
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
            ($env_type:ty, $env_config:expr, $agent_builder:ty) => {{
                let env: Box<$env_type> = Box::new($env_config.build_env(env_seed)?);
                let agent: Box<dyn Agent<_, _>> =
                    <$agent_builder>::new(agent_def).build_agent(&env, agent_seed)?;
                let log_hook = StepLogger::new(env.observation_space(), env.action_space());
                Box::new(Simulator::new(env, agent, (log_hook, hook)))
            }};
        }

        use EnvDef::*;
        Ok(match self {
            FixedMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    boxed_simulation!(
                        PomdpEnv<Bandit<Deterministic<f64>>>,
                        config,
                        ForFiniteFinite<_>
                    )
                }
                DistributionType::Bernoulli => {
                    boxed_simulation!(PomdpEnv<Bandit<Bernoulli>>, config, ForFiniteFinite<_>)
                }
            },
            UniformMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    boxed_simulation!(
                        PomdpEnv<Bandit<Deterministic<f64>>>,
                        config,
                        ForFiniteFinite<_>
                    )
                }
                DistributionType::Bernoulli => {
                    boxed_simulation!(PomdpEnv<Bandit<Bernoulli>>, config, ForFiniteFinite<_>)
                }
            },
            Chain(config) => {
                boxed_simulation!(PomdpEnv<ChainEnv>, config, ForFiniteFinite<_>)
            }
            MemoryGame(config) => {
                boxed_simulation!(PomdpEnv<MemoryGameEnv>, config, ForFiniteFinite<_>)
            }
            MetaOneHotBandits(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<OneHotBandits>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
            MetaUniformBernoulliBandits(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<UniformBernoulliBandits>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
            MetaDirichletMdps(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<Wrapped<DirichletRandomMdps, StepLimit>>,
                    config,
                    ForMetaFiniteFinite<_>
                )
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
            ($env_type:ty, $env_config:expr, $agent_builder:ty) => {{
                let env: $env_type = $env_config.build_env(env_seed)?;
                let agent: Box<dyn ManagerAgent<Worker = Box<dyn Agent<_, _> + Send>>> =
                    <$agent_builder>::new(agent_def).build_agent(&env, agent_seed)?;
                let log_hook = StepLogger::new(env.observation_space(), env.action_space());
                sim_config.build_simulator::<_, $env_type, _, _>(
                    $env_config.clone(),
                    agent,
                    (log_hook, hook),
                )
            }};
        }

        use EnvDef::*;
        Ok(match self {
            FixedMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    boxed_simulation!(
                        PomdpEnv<Bandit<Deterministic<f64>>>,
                        config,
                        ForFiniteFinite<_>
                    )
                }
                DistributionType::Bernoulli => {
                    boxed_simulation!(PomdpEnv<Bandit<Bernoulli>>, config, ForFiniteFinite<_>)
                }
            },
            UniformMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    boxed_simulation!(
                        PomdpEnv<Bandit<Deterministic<f64>>>,
                        config,
                        ForFiniteFinite<_>
                    )
                }
                DistributionType::Bernoulli => {
                    boxed_simulation!(PomdpEnv<Bandit<Bernoulli>>, config, ForFiniteFinite<_>)
                }
            },
            Chain(config) => {
                boxed_simulation!(PomdpEnv<ChainEnv>, config, ForFiniteFinite<_>)
            }
            MemoryGame(config) => {
                boxed_simulation!(PomdpEnv<MemoryGameEnv>, config, ForFiniteFinite<_>)
            }
            MetaOneHotBandits(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<OneHotBandits>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
            MetaUniformBernoulliBandits(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<UniformBernoulliBandits>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
            MetaDirichletMdps(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<Wrapped<DirichletRandomMdps, StepLimit>>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
        })
    }
}
