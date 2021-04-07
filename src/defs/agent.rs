use super::{OptimizerDef, PolicyDef};
use crate::agents::{
    Agent, AgentBuilder, BetaThompsonSamplingAgentConfig, BuildAgentError, PolicyGradientAgent,
    RandomAgentConfig, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::envs::EnvStructure;
use crate::spaces::{FeatureSpace, FiniteSpace, ParameterizedSampleSpace, RLSpace};
use std::fmt::Debug;
use tch::Tensor;

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
    /// A policy-gradient agent.
    PolicyGradient(PolicyGradientAgentDef),
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
            PolicyGradient(pg_def) => pg_def.build(es, seed),
            _ => Err(BuildAgentError::InvalidSpaceBounds),
        }
    }
}

/// Definition of a policy-gradient agent
#[derive(Debug)]
pub struct PolicyGradientAgentDef {
    pub steps_per_epoch: usize,
    pub policy: PolicyDef,
    pub optimizer: OptimizerDef,
}

impl Default for PolicyGradientAgentDef {
    fn default() -> Self {
        Self {
            steps_per_epoch: 1000,
            policy: Default::default(),
            optimizer: Default::default(),
        }
    }
}

impl<OS, AS> AgentBuilder<OS, AS> for PolicyGradientAgentDef
where
    OS: FeatureSpace<Tensor> + 'static,
    AS: ParameterizedSampleSpace<Tensor> + 'static,
{
    type Agent = Box<dyn Agent<OS::Element, AS::Element>>;

    fn build(&self, es: EnvStructure<OS, AS>, _seed: u64) -> Result<Self::Agent, BuildAgentError> {
        use PolicyDef::*;
        Ok(match &self.policy {
            Mlp(config) => Box::new(PolicyGradientAgent::new(
                es.observation_space,
                es.action_space,
                es.discount_factor,
                self.steps_per_epoch,
                config,
                &self.optimizer,
            )),
            GruMlp(config) => Box::new(PolicyGradientAgent::new(
                es.observation_space,
                es.action_space,
                es.discount_factor,
                self.steps_per_epoch,
                config,
                &self.optimizer,
            )),
            LstmMlp(config) => Box::new(PolicyGradientAgent::new(
                es.observation_space,
                es.action_space,
                es.discount_factor,
                self.steps_per_epoch,
                config,
                &self.optimizer,
            )),
        })
    }
}
