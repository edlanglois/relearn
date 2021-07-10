use super::{Options, Update, WithUpdate};
use crate::defs::{CriticUpdaterDef, PolicyUpdaterDef};
use crate::torch::updaters::{
    CriticLossUpdateRule, PolicyGradientUpdateRule, PpoPolicyUpdateRule, TrpoPolicyUpdateRule,
};

impl From<&Options> for PolicyUpdaterDef {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for PolicyUpdaterDef {
    fn update(&mut self, opts: &Options) {
        use PolicyUpdaterDef::*;
        match self {
            PolicyGradient(update_rule, optimizer_def) => {
                update_rule.update(opts);
                optimizer_def.update(opts);
            }
            Trpo(update_rule, optimizer_def) => {
                update_rule.update(opts);
                optimizer_def.update(opts);
            }
            Ppo(update_rule, optimizer_def) => {
                update_rule.update(opts);
                optimizer_def.update(opts);
            }
        }
    }
}

impl From<&Options> for PolicyGradientUpdateRule {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for PolicyGradientUpdateRule {
    fn update(&mut self, _: &Options) {}
}

impl From<&Options> for TrpoPolicyUpdateRule {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for TrpoPolicyUpdateRule {
    fn update(&mut self, opts: &Options) {
        if let Some(max_policy_step_kl) = opts.max_policy_step_kl {
            self.max_policy_step_kl = max_policy_step_kl;
        }
    }
}

impl From<&Options> for PpoPolicyUpdateRule {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for PpoPolicyUpdateRule {
    fn update(&mut self, opts: &Options) {
        if let Some(num_epochs) = opts.policy_epochs {
            self.num_epochs = num_epochs;
        }
        if let Some(clip_distance) = opts.ppo_clip_distance {
            self.clip_distance = clip_distance;
        }
    }
}

impl From<&Options> for CriticUpdaterDef {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for CriticUpdaterDef {
    fn update(&mut self, opts: &Options) {
        use CriticUpdaterDef::*;
        match self {
            CriticLoss(update_rule, optimizer_def) => {
                update_rule.update(opts);
                optimizer_def.update(opts);
            }
        }
    }
}

impl From<&Options> for CriticLossUpdateRule {
    fn from(opts: &Options) -> Self {
        Self::default().with_update(opts)
    }
}

impl Update<&Options> for CriticLossUpdateRule {
    fn update(&mut self, opts: &Options) {
        if let Some(optimizer_iters) = opts.critic_opt_iters {
            self.optimizer_iters = optimizer_iters;
        }
    }
}
