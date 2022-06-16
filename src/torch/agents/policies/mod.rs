//! Policies for an actor-critic agent.
mod actor;
mod ppo;
mod reinforce;
mod trpo;

pub use actor::PolicyActor;
pub use ppo::{Ppo, PpoConfig};
pub use reinforce::{Reinforce, ReinforceConfig};
pub use trpo::{Trpo, TrpoConfig};

use super::features::HistoryFeatures;
use super::WithCpuCopy;
use crate::logging::StatsLogger;
use crate::spaces::{NonEmptyFeatures, ParameterizedDistributionSpace};
use crate::torch::modules::{AsModule, Module, SeqIterative, SeqPacked};
use crate::torch::packed::PackedTensor;
use tch::{Device, Tensor};

/// A policy for an [actor-critic agent][super::ActorCriticAgent].
pub trait Policy: AsModule<Module = Self::PolicyModule> {
    type PolicyModule: Module + SeqPacked + SeqIterative;

    /// Update the policy module.
    ///
    /// # Args
    /// * `features`     - Experience features.
    /// * `advantages`   - Selected action values with a state baseline. Corresponds to `features`.
    ///                    May depend on the future within an episode.
    ///                    Appropriate for a REINFORCE-style policy gradient.
    /// * `action_space` - Environment action space.
    /// * `logger`       - Statistics logger.
    fn update<AS: ParameterizedDistributionSpace<Tensor> + ?Sized>(
        &mut self,
        features: &dyn HistoryFeatures,
        advantages: PackedTensor,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    );

    /// Create an actor for the policy module.
    fn actor<OS, AS>(
        &self,
        observation_space: NonEmptyFeatures<OS>,
        action_space: AS,
    ) -> PolicyActor<OS, AS, Self::Module> {
        PolicyActor::new(
            observation_space,
            action_space,
            self.as_module().shallow_clone(),
        )
    }
}

pub trait BuildPolicy {
    type Policy: Policy;

    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy;
}

impl<P: Policy> Policy for WithCpuCopy<P> {
    type PolicyModule = P::PolicyModule;

    fn update<AS: ParameterizedDistributionSpace<Tensor> + ?Sized>(
        &mut self,
        features: &dyn HistoryFeatures,
        advantages: PackedTensor,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) {
        self.as_inner_mut()
            .update(features, advantages, action_space, logger)
    }

    fn actor<OS, AS>(
        &self,
        observation_space: NonEmptyFeatures<OS>,
        action_space: AS,
    ) -> PolicyActor<OS, AS, Self::Module> {
        PolicyActor::new(
            observation_space,
            action_space,
            self.shallow_clone_module_cpu(),
        )
    }
}
