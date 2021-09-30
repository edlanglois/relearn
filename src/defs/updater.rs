use super::OptimizerDef;
use crate::spaces::ParameterizedDistributionSpace;
use crate::torch::optimizers::{BuildOptimizer, ConjugateGradientOptimizerConfig};
use crate::torch::updaters::{
    BuildCriticUpdater, BuildPolicyUpdater, CriticLossUpdateRule, PolicyGradientUpdateRule,
    PpoPolicyUpdateRule, TrpoPolicyUpdateRule, UpdateCritic, UpdatePolicy, WithOptimizer,
};
use tch::{nn::VarStore, Tensor};

/// Torch policy updater definition
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolicyUpdaterDef {
    PolicyGradient(PolicyGradientUpdateRule, OptimizerDef),
    Trpo(TrpoPolicyUpdateRule, ConjugateGradientOptimizerConfig),
    Ppo(PpoPolicyUpdateRule, OptimizerDef),
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
    pub fn default_ppo() -> Self {
        Self::Ppo(PpoPolicyUpdateRule::default(), OptimizerDef::default())
    }
}

impl<AS> BuildPolicyUpdater<AS> for PolicyUpdaterDef
where
    AS: ParameterizedDistributionSpace<Tensor> + ?Sized,
{
    type Updater = Box<dyn UpdatePolicy<AS> + Send>;

    fn build_policy_updater(&self, vs: &VarStore) -> Self::Updater {
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
            Ppo(update_rule, optimizer_def) => Box::new(WithOptimizer {
                update_rule: *update_rule,
                optimizer: optimizer_def.build_optimizer(vs).unwrap(),
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

impl BuildCriticUpdater for CriticUpdaterDef {
    type Updater = Box<dyn UpdateCritic + Send>;

    fn build_critic_updater(&self, vs: &VarStore) -> Self::Updater {
        use CriticUpdaterDef::*;
        match self {
            CriticLoss(update_rule, optimizer_def) => Box::new(WithOptimizer {
                update_rule: *update_rule,
                optimizer: optimizer_def.build_optimizer(vs).unwrap(),
            }),
        }
    }
}
