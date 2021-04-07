use super::Opts;
use crate::agents::{
    BetaThompsonSamplingAgentConfig, TabularQLearningAgentConfig, UCB1AgentConfig,
};
use crate::defs::{AgentDef, OptimizerDef, PolicyGradientAgentDef};
use clap::Clap;

/// Agent name
#[derive(Clap, Debug)]
pub enum AgentName {
    Random,
    TabularQLearning,
    BetaThompsonSampling,
    UCB1,
    MlpPolicyGradient,
}

impl From<&Opts> for AgentDef {
    fn from(opts: &Opts) -> Self {
        use AgentName::*;
        match opts.agent {
            Random => AgentDef::Random,
            TabularQLearning => AgentDef::TabularQLearning(TabularQLearningAgentConfig {
                exploration_rate: opts.exploration_rate,
            }),
            BetaThompsonSampling => {
                AgentDef::BetaThompsonSampling(BetaThompsonSamplingAgentConfig {
                    num_samples: opts.num_samples,
                })
            }
            UCB1 => AgentDef::UCB1(UCB1AgentConfig {
                exploration_rate: opts.exploration_rate,
            }),
            MlpPolicyGradient => AgentDef::PolicyGradient(PolicyGradientAgentDef {
                steps_per_epoch: opts.steps_per_epoch,
                policy: Default::default(), // TODO: set from opts,
                optimizer: OptimizerDef::default().with_learning_rate(opts.learning_rate),
            }),
        }
    }
}
