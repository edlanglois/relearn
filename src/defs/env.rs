use super::agent::{ForFiniteFinite, ForMetaFiniteFinite};
use super::AgentDef;
use crate::agents::{Agent, AgentBuilder};
use crate::envs::{
    Bandit, Chain as ChainEnv, DirichletRandomMdps, EnvBuilder, EnvWithState,
    FixedMeansBanditConfig, MemoryGame as MemoryGameEnv, MetaEnvConfig, OneHotBandits,
    PriorMeansBanditConfig, StatefulEnvironment, StatefulMetaEnv, StepLimit,
    UniformBernoulliBandits, WithState, Wrapped,
};
use crate::error::RLError;
use crate::logging::{Loggable, TimeSeriesLogger};
use crate::simulation::{
    hooks::StepLogger, GenericSimulationHook, RunSimulation, SimulationHook, Simulator,
};
use crate::spaces::{ElementRefInto, Space};
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
    /// * `logger` - A logger used to log the simulation statistics. Pass () for no logging.
    pub fn build_simulation<H, L>(
        &self,
        agent_def: &AgentDef,
        env_seed: u64,
        agent_seed: u64,
        hook: H,
        logger: L,
    ) -> Result<Box<dyn RunSimulation>, RLError>
    where
        L: TimeSeriesLogger + 'static,
        H: GenericSimulationHook + 'static,
    {
        /// Construct a boxed agent-environment simulation
        macro_rules! boxed_simulation {
            ($env_type:ty, $env_config:expr, $agent_builder:ty) => {{
                let env: Box<$env_type> = Box::new($env_config.build_env(env_seed)?);
                // TODO: Box the agent too?
                let agent = <$agent_builder>::new(agent_def).build_agent(&env, agent_seed)?;
                logging_boxed_simulator(env, agent, hook, logger)
            }};
        }

        use EnvDef::*;
        Ok(match self {
            FixedMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    boxed_simulation!(
                        EnvWithState<Bandit<Deterministic<f64>>>,
                        config,
                        ForFiniteFinite<_>
                    )
                }
                DistributionType::Bernoulli => {
                    boxed_simulation!(EnvWithState<Bandit<Bernoulli>>, config, ForFiniteFinite<_>)
                }
            },
            UniformMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    boxed_simulation!(
                        EnvWithState<Bandit<Deterministic<f64>>>,
                        config,
                        ForFiniteFinite<_>
                    )
                }
                DistributionType::Bernoulli => {
                    boxed_simulation!(EnvWithState<Bandit<Bernoulli>>, config, ForFiniteFinite<_>)
                }
            },
            Chain(config) => {
                boxed_simulation!(EnvWithState<ChainEnv>, config, ForFiniteFinite<_>)
            }
            MemoryGame(config) => {
                boxed_simulation!(EnvWithState<MemoryGameEnv>, config, ForFiniteFinite<_>)
            }
            MetaOneHotBandits(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<Wrapped<OneHotBandits, WithState>>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
            MetaUniformBernoulliBandits(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<Wrapped<UniformBernoulliBandits, WithState>>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
            MetaDirichletMdps(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<Wrapped<Wrapped<DirichletRandomMdps, StepLimit>, WithState>>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
        })
    }
}

/// Make a boxed simulator with an extra logging hook.
fn logging_boxed_simulator<OS, AS, H, L>(
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent: Box<dyn Agent<OS::Element, AS::Element>>,
    hook: H,
    logger: L,
) -> Box<dyn RunSimulation>
where
    OS: Space + ElementRefInto<Loggable> + 'static,
    <OS as Space>::Element: Clone,
    AS: Space + ElementRefInto<Loggable> + 'static,
    H: SimulationHook<OS::Element, AS::Element> + 'static,
    L: TimeSeriesLogger + 'static,
{
    let log_hook = StepLogger::new(environment.observation_space(), environment.action_space());
    Box::new(Simulator::new(environment, agent, (log_hook, hook), logger))
}
