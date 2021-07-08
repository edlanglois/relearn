use super::OptimizerDef;
use crate::spaces::ParameterizedDistributionSpace;
use crate::torch::backends::CudnnSupport;
use crate::torch::critic::Critic;
use crate::torch::optimizers::{ConjugateGradientOptimizerConfig, OptimizerBuilder};
use crate::torch::seq_modules::SequenceModule;
use crate::torch::updaters::{
    CriticLossUpdateRule, PolicyGradientUpdateRule, TrpoPolicyUpdateRule, UpdateCritic,
    UpdatePolicy, UpdaterBuilder, WithOptimizer,
};
use tch::{nn::VarStore, Tensor};

/// Torch policy updater definition
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolicyUpdaterDef {
    PolicyGradient(PolicyGradientUpdateRule, OptimizerDef),
    Trpo(TrpoPolicyUpdateRule, ConjugateGradientOptimizerConfig),
}

impl Default for PolicyUpdaterDef {
    fn default() -> Self {
        Self::default_policy_gradient()
    }
}

impl PolicyUpdaterDef {
    pub fn default_policy_gradient() -> Self {
        Self::PolicyGradient(PolicyGradientUpdateRule::default(), OptimizerDef::default())
    }
    pub fn default_trpo() -> Self {
        Self::Trpo(
            TrpoPolicyUpdateRule::default(),
            ConjugateGradientOptimizerConfig::default(),
        )
    }
}

impl<P, C, AS> UpdaterBuilder<Box<dyn UpdatePolicy<P, C, AS>>> for PolicyUpdaterDef
where
    P: SequenceModule + CudnnSupport + ?Sized,
    C: Critic + ?Sized,
    AS: ParameterizedDistributionSpace<Tensor> + ?Sized,
{
    fn build_updater(&self, vs: &VarStore) -> Box<dyn UpdatePolicy<P, C, AS>> {
        use PolicyUpdaterDef::*;
        match self {
            PolicyGradient(update_rule, optimizer_def) => Box::new(WithOptimizer {
                update_rule: *update_rule,
                optimizer: optimizer_def.build_optimizer(vs).unwrap(),
            }),
            Trpo(update_rule, cg_optimizer_config) => Box::new(WithOptimizer {
                update_rule: *update_rule,
                optimizer: cg_optimizer_config.build_optimizer(vs).unwrap(),
            }),
        }
    }
}

/// Torch critic updater definition
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CriticUpdaterDef {
    CriticLoss(CriticLossUpdateRule, OptimizerDef),
}

impl Default for CriticUpdaterDef {
    fn default() -> Self {
        Self::CriticLoss(CriticLossUpdateRule::default(), OptimizerDef::default())
    }
}

impl<C> UpdaterBuilder<Box<dyn UpdateCritic<C>>> for CriticUpdaterDef
where
    C: Critic + ?Sized,
{
    fn build_updater(&self, vs: &VarStore) -> Box<dyn UpdateCritic<C>> {
        use CriticUpdaterDef::*;
        match self {
            CriticLoss(update_rule, optimizer_def) => Box::new(WithOptimizer {
                update_rule: *update_rule,
                optimizer: optimizer_def.build_optimizer(vs).unwrap(),
            }),
        }
    }
}
