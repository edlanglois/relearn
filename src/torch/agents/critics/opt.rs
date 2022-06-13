use super::super::{n_backward_steps, ToLog};
use super::{
    AdvantageFn, BuildCritic, Critic, Device, HistoryFeatures, PackedTensor, SeqPacked,
    StateValueTarget, StatsLogger,
};
use crate::torch::modules::{BuildModule, Module};
use crate::torch::optimizers::{AdamConfig, BuildOptimizer, Optimizer};
use serde::{Deserialize, Serialize};
use tch::{COptimizer, Reduction};

/// Configuration for [`ValuesOpt`]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValuesOptConfig<MB, OC = AdamConfig> {
    /// Configuration for the state value function module.
    pub state_value_fn_config: MB,
    /// Configuration for the state value function module optimizer.
    pub optimizer_config: OC,
    /// Strategy for calculating advantage estimates given a state value function module.
    pub advantage_fn: AdvantageFn,
    /// Strategy for calculating state value target values.
    ///
    /// The state value module is updated to minimize its mean-squared-error to these targets.
    pub target: StateValueTarget,
    /// Number of optimization steps per update.
    ///
    /// ## Design Note
    /// Could be called `num_epochs` by analogy to supervised learning as the number of passes
    /// through the dataset in which the dataset is collected experience since the last agent
    /// update. However, the term "epoch" is used inconsistently in reinforcement learning,
    /// sometimes referring to an iteration of the collect-data-then-update-agent loop.
    pub opt_steps_per_update: u64,
    /// Upper bound on the environment discount factor.
    ///
    /// Effectively sets a maximum horizon on the number of steps of future reward considered.
    /// Low values bias the value estimates but reduce variance.
    pub max_discount_factor: f64,
}

impl<MB: Default, OC: Default> Default for ValuesOptConfig<MB, OC> {
    fn default() -> Self {
        Self {
            state_value_fn_config: MB::default(),
            optimizer_config: OC::default(),
            advantage_fn: AdvantageFn::default(),
            target: StateValueTarget::default(),
            opt_steps_per_update: 80,
            max_discount_factor: 0.99,
        }
    }
}

impl<MB, OC> BuildCritic for ValuesOptConfig<MB, OC>
where
    MB: BuildModule,
    MB::Module: SeqPacked,
    OC: BuildOptimizer,
    OC::Optimizer: Optimizer,
{
    type Critic = ValuesOpt<MB::Module, OC::Optimizer>;

    #[allow(clippy::cast_possible_truncation)]
    fn build_critic(&self, in_dim: usize, discount_factor: f64, device: Device) -> Self::Critic {
        let state_value_fn = self.state_value_fn_config.build_module(in_dim, 1, device);
        let optimizer = self
            .optimizer_config
            .build_optimizer(state_value_fn.trainable_variables())
            .unwrap();
        ValuesOpt {
            state_value_fn,
            optimizer,
            advantage_fn: self.advantage_fn,
            target: self.target,
            discount_factor: self.max_discount_factor.min(discount_factor) as f32,
            opt_steps_per_update: self.opt_steps_per_update,
        }
    }
}

/// Critic using a gradient-optimized state value function module.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValuesOpt<M, O = COptimizer> {
    state_value_fn: M,
    optimizer: O,
    advantage_fn: AdvantageFn,
    target: StateValueTarget,
    discount_factor: f32,
    opt_steps_per_update: u64,
}

impl<M, O> Critic for ValuesOpt<M, O>
where
    M: SeqPacked,
    O: Optimizer,
{
    fn advantages(&self, features: &dyn HistoryFeatures) -> PackedTensor {
        self.advantage_fn
            .advantages(&self.state_value_fn, self.discount_factor, features)
    }

    fn update(&mut self, features: &dyn HistoryFeatures, logger: &mut dyn StatsLogger) {
        let targets = self.target.targets(self.discount_factor, features);
        let observations = features.observation_features();
        let loss_fn = || {
            self.state_value_fn
                .seq_packed(observations)
                .tensor()
                .squeeze_dim(-1)
                .mse_loss(targets.tensor(), Reduction::Mean)
        };

        n_backward_steps(
            &mut self.optimizer,
            loss_fn,
            self.opt_steps_per_update,
            logger,
            ToLog::All,
            "critic update error",
        );
    }
}
