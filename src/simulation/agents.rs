//! Agent definitions
use crate::agents::{Agent, RandomAgent, TabularQLearningAgent};
use crate::envs::EnvStructure;
use crate::spaces::{FiniteSpace, Space};
use std::error::Error;
use std::fmt;

/// The definition of an agent
#[derive(Debug)]
pub enum AgentDef {
    /// An agent that selects actions randomly.
    Random,
    /// Epsilon-greedy tabular Q learning.
    TabularQLearning { exploration_rate: f32 },
}

impl AgentDef {
    /// Construct this agent for any observation and state space.
    pub fn make_any_any<OS, AS>(
        self,
        structure: EnvStructure<OS, AS>,
        seed: u64,
    ) -> Result<Box<dyn Agent<OS::Element, AS::Element>>, MakeAgentError>
    where
        OS: Space + fmt::Debug + 'static,
        AS: Space + fmt::Debug + 'static,
    {
        match self {
            AgentDef::Random => Ok(Box::new(RandomAgent::new(structure.action_space, seed))),
            _ => Err(MakeAgentError {
                agent: self,
                observation_space: Box::new(structure.observation_space),
                action_space: Box::new(structure.action_space),
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
        OS: FiniteSpace + fmt::Debug + 'static,
        AS: FiniteSpace + fmt::Debug + 'static,
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
            _ => self.make_any_any(structure, seed),
        }
    }
}

#[derive(Debug)]
pub struct MakeAgentError {
    agent: AgentDef,
    observation_space: Box<dyn fmt::Debug>,
    action_space: Box<dyn fmt::Debug>,
}

impl fmt::Display for MakeAgentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Agent {:?} does not support observation space {:?} and action space {:?}",
            self.agent, self.observation_space, self.action_space
        )
    }
}

impl Error for MakeAgentError {}
