//! Actor-critic agent
use super::critics::{BuildCritic, Critic};
use super::features::LazyHistoryFeatures;
use super::policies::{BuildPolicy, Policy, PolicyActor};
use super::WithCpuCopy;
use crate::agents::buffers::VecBuffer;
use crate::agents::{ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError, HistoryDataBound};
use crate::envs::EnvStructure;
use crate::logging::StatsLogger;
use crate::spaces::{FeatureSpace, NonEmptyFeatures, ParameterizedDistributionSpace};
use crate::torch::serialize::DeviceDef;
use crate::Prng;
use log::info;
use serde::{Deserialize, Serialize};
use tch::{Device, Tensor};

/// Configuration for [`ActorCriticAgent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorCriticConfig<PB, CB> {
    pub policy_config: PB,
    pub critic_config: CB,
    pub min_batch_size: HistoryDataBound,
    #[serde(with = "DeviceDef")]
    pub device: Device,
}

impl<PB, CB> Default for ActorCriticConfig<PB, CB>
where
    PB: Default,
    CB: Default,
{
    #[inline]
    fn default() -> Self {
        Self {
            policy_config: Default::default(),
            critic_config: Default::default(),
            min_batch_size: HistoryDataBound {
                min_steps: 10_000,
                slack_steps: 100,
            },
            device: Device::cuda_if_available(),
        }
    }
}

impl<OS, AS, PB, CB> BuildAgent<OS, AS> for ActorCriticConfig<PB, CB>
where
    OS: FeatureSpace + Clone,
    AS: ParameterizedDistributionSpace<Tensor> + Clone,
    PB: BuildPolicy,
    CB: BuildCritic,
{
    type Agent = ActorCriticAgent<OS, AS, PB::Policy, CB::Critic>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(ActorCriticAgent::new(env, self))
    }
}

/// Actor-crtic agent. Consists of a [`Policy`] and a [`Critic`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActorCriticAgent<OS, AS, P: Policy, C> {
    observation_space: NonEmptyFeatures<OS>,
    action_space: AS,

    policy: WithCpuCopy<P>,
    critic: C,
    min_batch_size: HistoryDataBound,
}

impl<OS, AS, P: Policy, C> ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
{
    pub fn new<E, PB, CB>(env: &E, config: &ActorCriticConfig<PB, CB>) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        PB: BuildPolicy<Policy = P>,
        CB: BuildCritic<Critic = C>,
    {
        let observation_space = NonEmptyFeatures::new(env.observation_space());
        let action_space = env.action_space();
        let num_observation_features = observation_space.num_features();

        let policy = config.policy_config.build_policy(
            num_observation_features,
            action_space.num_distribution_params(),
            config.device,
        );

        let critic = config.critic_config.build_critic(
            num_observation_features,
            env.discount_factor(),
            config.device,
        );

        Self {
            observation_space,
            action_space,
            policy: WithCpuCopy::new(policy, config.device),
            critic,
            min_batch_size: config.min_batch_size,
        }
    }
}

impl<OS, AS, P, C> Agent<OS::Element, AS::Element> for ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace + Clone,
    AS: ParameterizedDistributionSpace<Tensor> + Clone,
    P: Policy,
{
    type Actor = PolicyActor<OS, AS, P::Module>;

    fn actor(&self, _: ActorMode) -> Self::Actor {
        self.policy
            .actor(self.observation_space.clone(), self.action_space.clone())
    }
}

impl<OS, AS, P, C> BatchUpdate<OS::Element, AS::Element> for ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace,
    OS::Element: 'static,
    AS: ParameterizedDistributionSpace<Tensor>,
    AS::Element: 'static,
    P: Policy,
    C: Critic,
{
    type HistoryBuffer = VecBuffer<OS::Element, AS::Element>;

    fn buffer(&self) -> Self::HistoryBuffer {
        VecBuffer::with_capacity_for(self.min_batch_size)
    }

    fn min_update_size(&self) -> HistoryDataBound {
        self.min_batch_size
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        let mut buffers: Vec<_> = buffers.into_iter().collect();
        self.batch_update_slice(&mut buffers, logger);
    }
}

impl<OS, AS, P, C> ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: Policy,
    C: Critic,
{
    // Takes a slice of references because:
    // * It iterates over the buffers twice and it is awkward to make the right bounds for
    //      a "clone-able" (actually, into_iter with shorter lifetimes) generic iterator.
    // * The function is relatively large (if updates are inlined) and this avoids duplicate
    //      monomorphizations.
    // * Any inefficiency in the buffer access should be insignificant compared to the runtime
    //      cost of the rest of the update.
    /// Batch update given a slice of buffer references
    fn batch_update_slice(
        &mut self,
        buffers: &mut [&mut VecBuffer<OS::Element, AS::Element>],
        mut logger: &mut dyn StatsLogger,
    ) {
        let features = LazyHistoryFeatures::new(
            buffers.iter_mut().flat_map(|b| b.episodes()),
            &self.observation_space,
            &self.action_space,
            self.policy.device,
        );
        if features.is_empty() {
            info!("skipping model update; history buffer is empty");
            return;
        }

        let advantages =
            (&mut logger).log_elapsed("adv_est_time", |_| self.critic.advantages(&features));

        logger
            .with_scope("policy")
            .log_elapsed("update_time", |logger| {
                self.policy
                    .update(&features, advantages, &self.action_space, logger)
            });

        logger
            .with_scope("critic")
            .log_elapsed("update_time", |logger| {
                self.critic.update(&features, logger)
            });

        for buffer in buffers {
            buffer.clear()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::critics::{RewardToGoConfig, StepValueTarget, ValuesOptConfig};
    use super::super::policies::{PpoConfig, ReinforceConfig, TrpoConfig};
    use super::*;
    use crate::agents::testing;
    use crate::torch::modules::{BuildModule, GruMlpConfig, MlpConfig, SeqIterative, SeqPacked};
    use crate::torch::optimizers::AdamConfig;
    use rstest::rstest;
    use std::marker::PhantomData;

    trait FromModuleConfig<MB> {
        fn from_module_config(module_config: MB) -> Self;
    }

    impl<MB> FromModuleConfig<MB> for ReinforceConfig<MB> {
        fn from_module_config(module_config: MB) -> Self {
            Self {
                policy_fn_config: module_config,
                optimizer_config: AdamConfig {
                    learning_rate: 0.1,
                    ..AdamConfig::default()
                },
            }
        }
    }

    const fn reinforce<MB>() -> PhantomData<ReinforceConfig<MB>> {
        PhantomData
    }

    impl<MB: Default> FromModuleConfig<MB> for PpoConfig<MB> {
        fn from_module_config(module_config: MB) -> Self {
            Self {
                policy_fn_config: module_config,
                optimizer_config: AdamConfig {
                    learning_rate: 0.1,
                    ..AdamConfig::default()
                },
                opt_steps_per_update: 1,
                ..Self::default()
            }
        }
    }

    const fn ppo<MB>() -> PhantomData<ReinforceConfig<MB>> {
        PhantomData
    }

    impl<MB: Default> FromModuleConfig<MB> for TrpoConfig<MB> {
        fn from_module_config(module_config: MB) -> Self {
            Self {
                policy_fn_config: module_config,
                ..Self::default()
            }
        }
    }

    const fn trpo<MB>() -> PhantomData<ReinforceConfig<MB>> {
        PhantomData
    }

    fn values_opt_config<MB: Default>(
        module_config: MB,
        target: StepValueTarget,
    ) -> ValuesOptConfig<MB> {
        ValuesOptConfig {
            state_value_fn_config: module_config,
            optimizer_config: AdamConfig {
                learning_rate: 0.1,
                ..AdamConfig::default()
            },
            target,
            opt_steps_per_update: 1,
            ..ValuesOptConfig::default()
        }
    }

    #[rstest]
    #[allow(clippy::used_underscore_binding)] // confused by used of _policy_alg in macro expansion
    fn learns_deterministic_bandit_r2g<MB, PB>(
        #[values(MlpConfig::default(), GruMlpConfig::default())] policy_module: MB,
        #[values(reinforce(), ppo(), trpo())] _policy_alg: PhantomData<PB>,
        #[values(Device::Cpu, Device::cuda_if_available())] device: Device,
    ) where
        MB: BuildModule,
        MB::Module: SeqPacked + SeqIterative,
        PB: FromModuleConfig<MB> + BuildPolicy,
    {
        let config = ActorCriticConfig {
            policy_config: PB::from_module_config(policy_module),
            critic_config: RewardToGoConfig,
            min_batch_size: HistoryDataBound::new(25, 1),
            device,
        };
        testing::train_deterministic_bandit(&config, 10, 0.9);
    }

    #[rstest]
    #[allow(clippy::used_underscore_binding)] // confused by used of _policy_alg in macro expansion
    fn learns_deterministic_bandit_values_gae<MB, PB>(
        #[values(MlpConfig::default(), GruMlpConfig::default())] module: MB,
        #[values(reinforce(), ppo(), trpo())] _policy_alg: PhantomData<PB>,
        #[values(StepValueTarget::RewardToGo, StepValueTarget::OneStepTd)]
        value_target: StepValueTarget,
        #[values(Device::Cpu, Device::cuda_if_available())] device: Device,
    ) where
        MB: BuildModule + Default + Clone,
        MB::Module: SeqPacked + SeqIterative,
        PB: FromModuleConfig<MB> + BuildPolicy,
    {
        let config = ActorCriticConfig {
            policy_config: PB::from_module_config(module.clone()),
            critic_config: values_opt_config(module, value_target),
            min_batch_size: HistoryDataBound::new(25, 1),
            device,
        };
        testing::train_deterministic_bandit(&config, 10, 0.9);
    }
}
