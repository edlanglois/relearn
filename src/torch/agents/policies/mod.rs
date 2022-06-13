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
use crate::logging::StatsLogger;
use crate::spaces::{NonEmptyFeatures, ParameterizedDistributionSpace};
use crate::torch::modules::{Module, SeqIterative, SeqPacked};
use crate::torch::packed::PackedTensor;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use tch::{Device, Tensor};

/// A policy for an [actor-critic agent][super::ActorCriticAgent].
pub trait Policy {
    /// The policy function module.
    type Module: Module + SeqPacked + SeqIterative;

    /// Get a reference to the policy function module.
    fn policy_module(&self) -> &Self::Module;

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
            self.policy_module().shallow_clone(),
        )
    }
}

pub trait BuildPolicy {
    type Policy: Policy;

    fn build_policy(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Policy;
}

#[derive(Serialize, Deserialize)]
pub struct CpuActor<P: Policy> {
    pub policy: P,
    /// Device on which `policy` (the master copy) is stored.
    // Tensors will deserialize to CPU
    #[serde(skip, default = "cpu_device")]
    pub device: Device,
    #[serde(skip, default)]
    cpu_module: RefCell<Option<P::Module>>,
}

const fn cpu_device() -> Device {
    Device::Cpu
}

impl<P: Policy> CpuActor<P> {
    pub const fn new(policy: P, device: Device) -> Self {
        Self {
            policy,
            device,
            cpu_module: RefCell::new(None),
        }
    }
}

impl<P: Policy + Clone> Clone for CpuActor<P> {
    fn clone(&self) -> Self {
        Self::new(self.policy.clone(), self.device)
    }
}

impl<P: Policy + fmt::Debug> fmt::Debug for CpuActor<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("CpuActor")
            .field("policy", &self.policy)
            .field("device", &self.device)
            .field(
                "cpu_module",
                &self.cpu_module.borrow().as_ref().map(|_| "..."),
            )
            .finish()
    }
}

impl<P: Policy + PartialEq> PartialEq for CpuActor<P> {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.policy == other.policy
    }
}

impl<P: Policy> Policy for CpuActor<P> {
    type Module = P::Module;

    fn policy_module(&self) -> &Self::Module {
        self.policy.policy_module()
    }

    fn update<AS: ParameterizedDistributionSpace<Tensor> + ?Sized>(
        &mut self,
        features: &dyn HistoryFeatures,
        advantages: PackedTensor,
        action_space: &AS,
        logger: &mut dyn StatsLogger,
    ) {
        // The cpu module is a copy so it is invalidated when the original updates.
        self.cpu_module = RefCell::new(None);
        self.policy
            .update(features, advantages, action_space, logger)
    }

    fn actor<OS, AS>(
        &self,
        observation_space: NonEmptyFeatures<OS>,
        action_space: AS,
    ) -> PolicyActor<OS, AS, Self::Module> {
        let policy_module = if self.device == Device::Cpu {
            self.policy_module().shallow_clone()
        } else {
            self.cpu_module
                .borrow_mut()
                .get_or_insert_with(|| self.policy_module().clone_to_device(Device::Cpu))
                .shallow_clone()
        };
        PolicyActor::new(observation_space, action_space, policy_module)
    }
}
