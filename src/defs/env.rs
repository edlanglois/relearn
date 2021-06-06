use super::AgentDef;
use crate::agents::{Agent, BuildAgentError};
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
use crate::spaces::{ElementRefInto, FiniteSpace, RLActionSpace, RLObservationSpace, Space};
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
        use EnvDef::*;
        Ok(match self {
            FixedMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    let env: EnvWithState<Bandit<Deterministic<f64>>> =
                        config.build_env(env_seed)?;
                    finite_finite_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
                }
                DistributionType::Bernoulli => {
                    let env: EnvWithState<Bandit<Bernoulli>> = config.build_env(env_seed)?;
                    finite_finite_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
                }
            },
            UniformMeanBandit(dist_type, config) => match dist_type {
                DistributionType::Deterministic => {
                    let env: EnvWithState<Bandit<Deterministic<f64>>> =
                        config.build_env(env_seed)?;
                    finite_finite_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
                }
                DistributionType::Bernoulli => {
                    let env: EnvWithState<Bandit<Bernoulli>> = config.build_env(env_seed)?;
                    finite_finite_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
                }
            },
            Chain(config) => {
                let env: EnvWithState<ChainEnv> = config.build_env(env_seed)?;
                finite_finite_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
            }
            MemoryGame(config) => {
                let env: EnvWithState<MemoryGameEnv> = config.build_env(env_seed)?;
                finite_finite_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
            }
            MetaOneHotBandits(config) => {
                let env: StatefulMetaEnv<DistWithState<OneHotBandits>> =
                    config.build_env(env_seed)?;
                any_any_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
            }
            MetaUniformBernoulliBandits(config) => {
                let env: StatefulMetaEnv<DistWithState<UniformBernoulliBandits>> =
                    config.build_env(env_seed)?;
                any_any_simulator(Box::new(env), agent_def, logger, hook, agent_seed)?
            }
        })
    }
}

/// Make a boxed simulator for an environment with a finite observation and action space.
fn finite_finite_simulator<OS, AS, L, H>(
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent_def: &AgentDef,
    logger: L,
    hook: H,
    agent_seed: u64,
) -> Result<Box<dyn Simulation>, BuildAgentError>
where
    OS: RLObservationSpace + FiniteSpace + 'static,
    <OS as Space>::Element: Clone,
    AS: RLActionSpace + FiniteSpace + 'static,
    L: Logger + 'static,
    H: SimulationHook<OS::Element, AS::Element, L> + 'static,
{
    let agent = agent_def.build_finite_finite(&environment, agent_seed)?;
    Ok(logging_boxed_simulator(environment, agent, logger, hook))
}

/// Make a boxed simulator for an environment with any observation and action space.
///
/// Fails if the agent requires a more specialized bound on the observation or action spaces.
fn any_any_simulator<OS, AS, L, H>(
    environment: Box<dyn StatefulEnvironment<ObservationSpace = OS, ActionSpace = AS>>,
    agent_def: &AgentDef,
    logger: L,
    hook: H,
    agent_seed: u64,
) -> Result<Box<dyn Simulation>, BuildAgentError>
where
    OS: RLObservationSpace + 'static,
    <OS as Space>::Element: Clone,
    AS: RLActionSpace + 'static,
    L: Logger + 'static,
    H: SimulationHook<OS::Element, AS::Element, L> + 'static,
{
    let agent = agent_def.build_any_any(&environment, agent_seed)?;
    Ok(logging_boxed_simulator(environment, agent, logger, hook))
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
