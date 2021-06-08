use super::agent::{ForFiniteFinite, ForMetaFiniteFinite};
use super::AgentDef;
use crate::agents::{Agent, AgentBuilder};
use crate::envs::{
    Bandit, Chain as ChainEnv, DistWithState, EnvBuilder, EnvWithState, FixedMeansBanditConfig,
    MemoryGame as MemoryGameEnv, MetaEnvConfig, OneHotBandits, PriorMeansBanditConfig,
    StatefulEnvironment, StatefulMetaEnv, UniformBernoulliBandits,
};
use crate::error::RLError;
use crate::logging::{Loggable, Logger};
use crate::simulation::{
    hooks::StepLogger, GenericSimulationHook, Simulation, SimulationHook, Simulator,
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
    /// * `logger` - A logger used to log the simulation statistics. Pass () for no logging.
    /// * `hook` - A hook called on each step of the simulation. Pass () for no hook.
    pub fn build_simulation<L, H>(
        &self,
        agent_def: &AgentDef,
        env_seed: u64,
        agent_seed: u64,
        logger: L,
        hook: H,
    ) -> Result<Box<dyn Simulation>, RLError>
    where
        L: Logger + 'static,
        H: GenericSimulationHook + 'static,
    {
        /// Construct a boxed agent-environment simulation
        macro_rules! boxed_simulation {
            ($env_type:ty, $env_config:expr, $agent_builder:ty) => {{
                let env: Box<$env_type> = Box::new($env_config.build_env(env_seed)?);
                // TODO: Box the agent too?
                let agent = <$agent_builder>::new(agent_def).build_agent(&env, agent_seed)?;
                logging_boxed_simulator(env, agent, logger, hook)
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
                    StatefulMetaEnv<DistWithState<OneHotBandits>>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
            MetaUniformBernoulliBandits(config) => {
                boxed_simulation!(
                    StatefulMetaEnv<DistWithState<UniformBernoulliBandits>>,
                    config,
                    ForMetaFiniteFinite<_>
                )
            }
        })
    }
}

/// Make a boxed simulator with an extra logging hook.
fn logging_boxed_simulator<OS, AS, L, H>(
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent: Box<dyn Agent<OS::Element, AS::Element>>,
    logger: L,
    hook: H,
) -> Box<dyn Simulation>
where
    OS: Space + ElementRefInto<Loggable> + 'static,
    <OS as Space>::Element: Clone,
    AS: Space + ElementRefInto<Loggable> + 'static,
    L: Logger + 'static,
    H: SimulationHook<OS::Element, AS::Element, L> + 'static,
{
    let log_hook = StepLogger::new(environment.observation_space(), environment.action_space());
    Box::new(Simulator::new(environment, agent, logger, (log_hook, hook)))
}
