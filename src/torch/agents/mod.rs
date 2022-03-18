mod actor;
mod actor_critic;
pub mod critic;
pub mod features;
pub mod learning_critic;
pub mod learning_policy;
pub mod policy;
#[cfg(test)]
mod tests;

pub use actor::PolicyActor;
pub use actor_critic::{ActorCriticAgent, ActorCriticConfig};

use super::modules::Module;
use super::optimizers::BuildOptimizer;
use serde::{Deserialize, Serialize};

/// Optimize a module using an optimizer and an update rule.
///
/// Many learning policies and critics take this form.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RuleOpt<M, O, U> {
    /// Optimize the variables of this module.
    module: M,
    /// Optimizer: updates the trainable varaibles of `module` to minimize a given loss function.
    optimizer: O,
    /// Implements the update of `module` using `optimizer`.
    update_rule: U,
}

impl<M, O, U> RuleOpt<M, O, U>
where
    M: Module,
{
    #[inline]
    pub fn new<OB>(module: M, optimizer_config: &OB, update_rule: U) -> Self
    where
        OB: BuildOptimizer<Optimizer = O>,
    {
        Self {
            optimizer: optimizer_config
                .build_optimizer(module.trainable_variables())
                .unwrap(),
            module,
            update_rule,
        }
    }
}

/// Configuration for [`RuleOpt`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuleOptConfig<MB, OB, UB> {
    pub module_config: MB,
    pub optimizer_config: OB,
    pub update_rule_config: UB,
}
