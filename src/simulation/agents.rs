//! Agent definitions
use super::spaces::{CommonActionSpace, CommonObservationSpace};
use crate::agents::error::NewAgentError;
use crate::agents::{
    Agent, BetaThompsonSamplingAgent, PolicyGradientAgent, RandomAgent, TabularQLearningAgent,
    UCB1Agent,
};
use crate::envs::EnvStructure;
use crate::spaces::FiniteSpace;
use crate::torch::configs::MLPConfig;
use tch::nn::Adam;
use thiserror::Error;

/// The definition of an agent
#[derive(Debug)]
pub enum AgentDef {
    /// An agent that selects actions randomly.
    Random,
    /// Epsilon-greedy tabular Q learning.
    TabularQLearning { exploration_rate: f64 },
    /// Thompson sampling of for Bernoulli rewards using Beta priors.
    ///
    /// Assumes no relationship between states.
    BetaThompsonSampling { num_samples: usize },
    /// UCB1 agent from Auer 2002
    UCB1 { exploration_rate: f64 },
    /// A simple MLP policy gradient agent.
    SimpleMLPPolicyGradient {
        steps_per_epoch: usize,
        learning_rate: f64,
    },
}

impl AgentDef {
    /// Construct this agent for any observation and state space.
    pub fn make_any_any<OS, AS>(
        self,
        structure: EnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<Box<dyn Agent<OS::Element, AS::Element>>, MakeAgentError>
    where
        OS: CommonObservationSpace + 'static,
        AS: CommonActionSpace + 'static,
    {
        match self {
            AgentDef::Random => Ok(Box::new(RandomAgent::new(structure.action_space, seed))),
            _ => Err(MakeAgentError {
                agent: self,
                cause: NewAgentError::InvalidSpace {
                    observation_space: Box::new(structure.observation_space),
                    action_space: Box::new(structure.action_space),
                },
            }),
        }
    }

    /// Construct this agent for finite observation and action spaces.
    pub fn make_finite_finite<OS, AS>(
        self,
        structure: EnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<Box<dyn Agent<OS::Element, AS::Element>>, MakeAgentError>
    where
        OS: CommonObservationSpace + FiniteSpace + 'static,
        AS: CommonActionSpace + FiniteSpace + 'static,
    {
        match self {
            AgentDef::TabularQLearning { exploration_rate } => {
                Ok(Box::new(TabularQLearningAgent::new(
                    structure.observation_space,
                    structure.action_space,
                    structure.discount_factor,
                    exploration_rate,
                    seed,
                )))
            }
            AgentDef::BetaThompsonSampling { num_samples } => {
                Ok(Box::new(BetaThompsonSamplingAgent::new(
                    structure.observation_space,
                    structure.action_space,
                    structure.reward_range,
                    num_samples,
                    seed,
                )))
            }
            AgentDef::UCB1 { exploration_rate } => {
                match UCB1Agent::new(
                    structure.observation_space,
                    structure.action_space,
                    structure.reward_range,
                    exploration_rate,
                ) {
                    Ok(agent) => Ok(Box::new(agent)),
                    Err(cause) => Err(MakeAgentError { agent: self, cause }),
                }
            }
            AgentDef::SimpleMLPPolicyGradient {
                steps_per_epoch,
                learning_rate,
            } => Ok(Box::new(PolicyGradientAgent::new(
                structure.observation_space,
                structure.action_space,
                structure.discount_factor,
                steps_per_epoch,
                learning_rate,
                &MLPConfig::default(),
                Adam::default(),
            ))),
            _ => self.make_any_any(structure, seed),
        }
    }
}

#[derive(Debug, Error)]
#[error("error constructing agent {agent:?}")]
pub struct MakeAgentError {
    agent: AgentDef,
    #[source]
    cause: NewAgentError,
}
