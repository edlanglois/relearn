use super::{
    BuildPolicy, HistoryFeatures, Module, PackedTensor, ParameterizedDistributionSpace, Policy,
    SeqIterative, SeqPacked, StatsLogger,
};
use crate::torch::backends::WithCudnnEnabled;
use crate::torch::modules::BuildModule;
use crate::torch::optimizers::{
    BuildOptimizer, ConjugateGradientOptimizer, ConjugateGradientOptimizerConfig,
    OptimizerStepError, TrustRegionOptimizer,
};
use crate::utils::distributions::ArrayDistribution;
use log::warn;
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

/// Configuration for [`Trpo`]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrpoConfig<MB, OC = ConjugateGradientOptimizerConfig> {
    pub policy_fn_config: MB,
    pub optimizer_config: OC,
    /// Maximum policy KL divergence when taking a step.
    ///
    /// Specifically, this is the mean KL divergence of the action distributions across all
    /// observed states.
    pub max_policy_step_kl: f64,
}

impl<MB, OC> Default for TrpoConfig<MB, OC>
where
    MB: Default,
    OC: Default,
{
    fn default() -> Self {
        Self {
            policy_fn_config: MB::default(),
            optimizer_config: OC::default(),
            // This step size was used by all experiments in Schulman's TRPO paper.
            max_policy_step_kl: 0.01,
        }
    }
}

impl<MB, OC> BuildPolicy for TrpoConfig<MB, OC>
where
    MB: BuildModule,
    MB::Module: SeqPacked + SeqIterative,
    OC: BuildOptimizer,
    OC::Optimizer: TrustRegionOptimizer,
{
    type Policy = Trpo<MB::Module, OC::Optimizer>;

    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy {
        let policy_fn = self.policy_fn_config.build_module(in_dim, out_dim, device);
        let optimizer = self
            .optimizer_config
            .build_optimizer(policy_fn.trainable_variables())
            .unwrap();
        Trpo {
            policy_fn,
            optimizer,
            max_policy_step_kl: self.max_policy_step_kl,
        }
    }
}

/// Trust Region Policy Optimization (TRPO) with a clipped objective.
///
/// # Reference
/// "[Trust Region Policy Optimization][trpo]" by Schulman et al.
///
/// [trpo]: https://arxiv.org/abs/1502.05477
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trpo<M, O = ConjugateGradientOptimizer> {
    policy_fn: M,
    optimizer: O,
    /// Maximum policy KL-divergence per update
    max_policy_step_kl: f64,
}

impl<M, O> Policy for Trpo<M, O>
where
    M: Module + SeqPacked + SeqIterative,
    O: TrustRegionOptimizer,
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
        let _cudnn_disable_guard = if self.policy_fn.has_cudnn_second_derivatives() {
            None
        } else {
            Some(WithCudnnEnabled::new(false))
        };
        let observation_features = features.observation_features();
        let actions = features.actions().tensor();

        let (initial_distribution, initial_log_probs) = {
            let _no_grad = tch::no_grad_guard();

            let policy_output = self.policy_fn.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());
            let log_probs = distribution.log_probs(actions);
            let entropy = distribution.entropy().mean(Kind::Float);
            logger.log_scalar("entropy", entropy.into());

            (distribution, log_probs)
        };

        let mut policy_loss_distance_fn = || {
            let policy_output = self.policy_fn.seq_packed(observation_features);
            let distribution = action_space.distribution(policy_output.tensor());

            let log_probs = distribution.log_probs(actions);
            let likelihood_ratio = (log_probs - &initial_log_probs).exp();
            let loss = -(likelihood_ratio * advantages.tensor()).mean(Kind::Float);

            // NOTE:
            // The [TRPO paper] and [Garage] use `KL(old_policy || new_policy)` while
            // [Spinning Up] uses `KL(new_policy || old_policy)`.
            //
            // I do not know why Spinning Up differs. I follow the TRPO paper and Garage.
            //
            // [TRPO paper]: <https://arxiv.org/abs/1502.05477>
            // [Garage]: <https://garage.readthedocs.io/en/latest/user/algo_trpo.html>
            // [Spinning Up]: <https://spinningup.openai.com/en/latest/algorithms/trpo.html>
            let distance = initial_distribution
                .kl_divergence_from(&distribution)
                .mean(Kind::Float);

            (loss, distance)
        };

        let result = self.optimizer.trust_region_backward_step(
            &mut policy_loss_distance_fn,
            self.max_policy_step_kl,
            logger,
        );

        if let Err(error) = result {
            match error {
                OptimizerStepError::NaNLoss => panic!("NaN loss in policy optimization"),
                OptimizerStepError::NaNConstraint => {
                    panic!("NaN constraint in policy optimization")
                }
                err => warn!("error in policy step: {}", err),
            };
        }
    }
}
