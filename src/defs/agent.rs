use super::{OptimizerDef, PolicyDef};
use crate::agents::{
    Agent, AgentBuilder, BetaThompsonSamplingAgentConfig, BuildAgentError,
    GaePolicyGradientAgentConfig, PolicyGradientAgentConfig, RandomAgentConfig,
    TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::envs::EnvStructure;
use crate::spaces::{FiniteSpace, RLSpace};
use crate::torch::configs::MlpConfig;
use std::fmt::Debug;

/// Agent definition
#[derive(Debug)]
pub enum AgentDef {
    /// An agent that selects actions randomly.
    Random,
    /// Epsilon-greedy tabular Q learning.
    TabularQLearning(TabularQLearningAgentConfig),
    /// Thompson sampling of for Bernoulli rewards using Beta priors.
    ///
    /// Assumes no relationship between states.
    BetaThompsonSampling(BetaThompsonSamplingAgentConfig),
    /// UCB1 agent from Auer 2002
    UCB1(UCB1AgentConfig),
    /// Policy gradient.
    PolicyGradient(PolicyGradientAgentConfig<PolicyDef, OptimizerDef>),
    /// Policy gradient with Generalized Advantage Estimation.
    GaePolicyGradient(
        GaePolicyGradientAgentConfig<PolicyDef, OptimizerDef, MlpConfig, OptimizerDef>,
    ),
}

// TODO: Return Box<dyn ActorAgent> where ActorAgent: Actor + Agent instead of Box<dyn Agent>

impl AgentDef {
    /// Construct an agent for the given environment structure.
    ///
    /// The observation and action spaces must both be finite.
    pub fn build_finite_finite<OS, AS>(
        &self,
        es: EnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<Box<dyn Agent<OS::Element, AS::Element>>, BuildAgentError>
    where
        OS: RLSpace + FiniteSpace + 'static,
        AS: RLSpace + FiniteSpace + 'static,
    {
        use AgentDef::*;
        match self {
            TabularQLearning(config) => config.build(es, seed).map(|a| Box::new(a) as _),
            BetaThompsonSampling(config) => config.build(es, seed).map(|a| Box::new(a) as _),
            UCB1(config) => config.build(es, seed).map(|a| Box::new(a) as _),
            _ => self.build_any_any(es, seed),
        }
    }

    /// Construct an agent for the given environment structure and generic spaces.
    ///
    /// There is no trait specialization so this will fail if the agent cannot be built for
    /// arbitrary spaces, even if it can for the specific instance this function is called with.
    pub fn build_any_any<OS, AS>(
        &self,
        es: EnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<Box<dyn Agent<OS::Element, AS::Element>>, BuildAgentError>
    where
        OS: RLSpace + 'static,
        AS: RLSpace + 'static,
    {
        use AgentDef::*;
        match self {
            Random => RandomAgentConfig::new()
                .build(es, seed)
                .map(|a| Box::new(a) as _),
            PolicyGradient(config) => config.build(es, seed).map(|a| Box::new(a) as _),
            GaePolicyGradient(config) => config.build(es, seed).map(|a| Box::new(a) as _),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
    }
}
