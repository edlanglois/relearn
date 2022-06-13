use super::super::{n_backward_steps, ToLog};
use super::{
    BuildPolicy, HistoryFeatures, Module, PackedTensor, ParameterizedDistributionSpace, Policy,
    SeqIterative, SeqPacked, StatsLogger,
};
use crate::torch::modules::BuildModule;
use crate::torch::optimizers::{AdamConfig, BuildOptimizer, Optimizer};
use crate::utils::distributions::ArrayDistribution;
use serde::{Deserialize, Serialize};
use tch::{COptimizer, Device, Kind, Tensor};

/// Configuration for [`Ppo`]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct PpoConfig<MB, OC = AdamConfig> {
    pub policy_fn_config: MB,
    pub optimizer_config: OC,
    /// Number of optimization steps per update.
    pub opt_steps_per_update: u64,
    // TODO: Support minibatches
    // pub minibatch_size: usize,
    /// Clip the surrogate objective to `1 ± clip_distance`.
    ///
    /// This is ε (epsilon) in the paper.
    pub clip_distance: f64,
}

impl<MB, OC> Default for PpoConfig<MB, OC>
where
    MB: Default,
    OC: Default,
{
    fn default() -> Self {
        Self {
            policy_fn_config: MB::default(),
            optimizer_config: OC::default(),
            opt_steps_per_update: 10,
            clip_distance: 0.2,
        }
    }
}

impl<MB, OC> BuildPolicy for PpoConfig<MB, OC>
where
    MB: BuildModule,
    MB::Module: SeqPacked + SeqIterative,
    OC: BuildOptimizer,
    OC::Optimizer: Optimizer,
{
    type Policy = Ppo<MB::Module, OC::Optimizer>;

    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy {
        let policy_fn = self.policy_fn_config.build_module(in_dim, out_dim, device);
        let optimizer = self
            .optimizer_config
            .build_optimizer(policy_fn.trainable_variables())
            .unwrap();
        Ppo {
            policy_fn,
            optimizer,
            opt_steps_per_update: self.opt_steps_per_update,
            clip_distance: self.clip_distance,
        }
    }
}

/// Proximal Policy Optimization (PPO) with a clipped objective.
///
/// # Reference
/// "[Proximal Policy Optimization Algorithms][ppo]" by Schulman et al.
///
/// [ppo]: https://arxiv.org/abs/1707.06347
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ppo<M, O = COptimizer> {
    policy_fn: M,
    optimizer: O,
    opt_steps_per_update: u64,
    clip_distance: f64,
}

impl<M, O> Policy for Ppo<M, O>
where
    M: Module + SeqPacked + SeqIterative,
    O: Optimizer,
{
    type Module = M;

    fn policy_module(&self) -> &Self::Module {
        &self.policy_fn
    }

    fn update<AS: ParameterizedDistributionSpace<Tensor> + ?Sized>(
        &mut self,
        features: &dyn HistoryFeatures,
        advantages: PackedTensor,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) {
        let observation_features = features.observation_features();
        let actions = features.actions().tensor();

        let initial_log_probs = {
            let _no_grad = tch::no_grad_guard();

            let policy_output = self.policy_fn.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());
            let log_probs = distribution.log_probs(actions);
            let entropy = distribution.entropy().mean(Kind::Float);
            logger.log_scalar("entropy", entropy.into());

            log_probs
        };

        // TODO Sample a minibatch on each update.
        let policy_surrogate_loss_fn = || {
            let policy_output = self.policy_fn.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());
            let log_probs = distribution.log_probs(actions);

            let likelihood_ratio = (log_probs - &initial_log_probs).exp();
            let clipped_likelihood_ratio =
                likelihood_ratio.clip(1.0 - self.clip_distance, 1.0 + self.clip_distance);

            (likelihood_ratio * advantages.tensor())
                .min_other(&(clipped_likelihood_ratio * advantages.tensor()))
                .mean(Kind::Float)
                .neg()
        };

        n_backward_steps(
            &mut self.optimizer,
            policy_surrogate_loss_fn,
            self.opt_steps_per_update,
            logger,
            ToLog::NoAbsLoss, // loss value is offset by a meaningless constant
            "policy update error",
        );
    }
}
