use super::{
    BuildPolicy, HistoryFeatures, Module, PackedTensor, ParameterizedDistributionSpace, Policy,
    SeqIterative, SeqPacked, StatsLogger,
};
use crate::torch::modules::BuildModule;
use crate::torch::optimizers::{AdamConfig, BuildOptimizer, Optimizer};
use crate::utils::distributions::ArrayDistribution;
use serde::{Deserialize, Serialize};
use tch::{COptimizer, Device, Kind, Tensor};

/// Configuration for [`Reinforce`]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReinforceConfig<MB, OC = AdamConfig> {
    pub policy_fn_config: MB,
    pub optimizer_config: OC,
}

impl<MB, OC> BuildPolicy for ReinforceConfig<MB, OC>
where
    MB: BuildModule,
    MB::Module: SeqPacked + SeqIterative,
    OC: BuildOptimizer,
    OC::Optimizer: Optimizer,
{
    type Policy = Reinforce<MB::Module, OC::Optimizer>;

    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy {
        let policy_fn = self.policy_fn_config.build_module(in_dim, out_dim, device);
        let optimizer = self
            .optimizer_config
            .build_optimizer(policy_fn.trainable_variables())
            .unwrap();
        Reinforce {
            policy_fn,
            optimizer,
        }
    }
}

/// REINFORCE policy gradient
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Reinforce<M, O = COptimizer> {
    policy_fn: M,
    optimizer: O,
}

impl<M, O> Policy for Reinforce<M, O>
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
        let mut entropies = None;
        let mut policy_loss_fn = || {
            let action_dist_params = self.policy_fn.seq_packed(features.observation_features());

            let action_distributions = action_space.distribution(action_dist_params.tensor());
            let log_probs = action_distributions.log_probs(features.actions().tensor());
            entropies.get_or_insert_with(|| action_distributions.entropy());
            -(log_probs * advantages.tensor()).mean(Kind::Float)
        };

        let _ = self
            .optimizer
            .backward_step(&mut policy_loss_fn, logger)
            .unwrap();

        if let Some(entropy) = entropies.map(|e| e.mean(Kind::Float).into()) {
            logger.log_scalar("entropy", entropy);
        }
    }
}
