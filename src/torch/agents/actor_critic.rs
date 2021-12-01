use super::super::{
    critic::{BuildCritic, Critic},
    policy::BuildPolicy,
    updaters::{BuildCriticUpdater, BuildPolicyUpdater},
};
use crate::agents::SetActorMode;
use crate::envs::EnvStructure;
use crate::spaces::{NonEmptyFeatures, NumFeatures, ParameterizedDistributionSpace, Space};
use tch::{nn::VarStore, Device, Tensor};

/// Configuration for [`ActorCriticAgent`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActorCriticConfig<PB, PUB, CB, CUB> {
    pub policy_config: PB,
    pub policy_updater_config: PUB,
    pub critic_config: CB,
    pub critic_updater_config: CUB,
    pub device: Device,
}

impl<PB, PUB, CB, CUB> Default for ActorCriticConfig<PB, PUB, CB, CUB>
where
    PB: Default,
    PUB: Default,
    CB: Default,
    CUB: Default,
{
    fn default() -> Self {
        Self {
            policy_config: Default::default(),
            policy_updater_config: Default::default(),
            critic_config: Default::default(),
            critic_updater_config: Default::default(),
            device: Device::Cpu,
        }
    }
}

/*
impl<PB, PUB, CB, CUB, OS, AS> BuildBatchUpdateActor<OS, AS> for ActorCriticConfig<PB, PUB, CB, CUB>
where
    PB: BuildPolicy,
    PUB: BuildPolicyUpdater<AS>,
    CB: BuildCritic,
    CUB: BuildCriticUpdater,
    OS: EncoderFeatureSpace,
    AS: ReprSpace<Tensor> + ParameterizedDistributionSpace<Tensor>,
{
    #[allow(clippy::type_complexity)]
    type BatchUpdateActor =
        ActorCriticAgent<OS, AS, PB::Policy, PUB::Updater, CB::Critic, CUB::Updater>;

    fn build_batch_update_actor(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _seed: u64,
    ) -> Result<Self::BatchUpdateActor, BuildAgentError> {
        Ok(ActorCriticAgent::new(env, self))
    }
}
*/

/// Actor-critic agent.
#[derive(Debug)]
pub struct ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: Space,
    AS: Space,
{
    /// Environment observation space
    pub observation_space: NonEmptyFeatures<OS>,

    /// Environment action space
    pub action_space: AS,

    /// Amount by which future rewards are discounted
    pub discount_factor: f64,

    /// Device on which model variables are stored.
    pub device: Device,

    /// The policy module (the "actor").
    pub policy: P,

    /// The policy updater.
    ///
    /// Performs a policy update given history data.
    pub policy_updater: PU,

    /// The primary policy variables.
    ///
    /// Used for updating `cpu_policy_variables`.
    policy_variables: VarStore,

    /// A copy of the policy on the CPU for faster actions.
    ///
    /// Exists if the primary policy is not already on the CPU.
    cpu_policy: Option<P>,

    /// The variables used by the CPU policy.
    ///
    /// Exists if cpu_policy exists.
    cpu_policy_variables: Option<VarStore>,

    /// The critic module.
    pub critic: C,

    /// The critic updater.
    ///
    /// Performs a critic update given history data.
    pub critic_updater: CU,
}

impl<OS, AS, P, PU, C, CU> ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: Space + NumFeatures,
    AS: ParameterizedDistributionSpace<Tensor>,
{
    pub fn new<E, PB, PUB, CB, CUB>(env: &E, config: &ActorCriticConfig<PB, PUB, CB, CUB>) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        PB: BuildPolicy<Policy = P>,
        PUB: BuildPolicyUpdater<AS, Updater = PU>,
        C: Critic,
        CB: BuildCritic<Critic = C>,
        CUB: BuildCriticUpdater<Updater = CU>,
    {
        let observation_space = NonEmptyFeatures::new(env.observation_space());
        let action_space = env.action_space();

        let policy_variables = VarStore::new(config.device);
        let policy = config.policy_config.build_policy(
            &policy_variables.root(),
            observation_space.num_features(),
            action_space.num_distribution_params(),
        );
        let policy_updater = config
            .policy_updater_config
            .build_policy_updater(&policy_variables);

        // A copy of the policy on the CPU for faster actions
        let (cpu_policy, cpu_policy_variables) = if config.device != Device::Cpu {
            let mut cpu_policy_variables = VarStore::new(Device::Cpu);
            let cpu_policy = config.policy_config.build_policy(
                &cpu_policy_variables.root(),
                observation_space.num_features(),
                action_space.num_distribution_params(),
            );
            cpu_policy_variables.copy(&policy_variables).unwrap();
            (Some(cpu_policy), Some(cpu_policy_variables))
        } else {
            (None, None)
        };

        let critic_variables = VarStore::new(config.device);
        let critic = config
            .critic_config
            .build_critic(&critic_variables.root(), observation_space.num_features());
        let discount_factor = critic.discount_factor(env.discount_factor());
        let critic_updater = config
            .critic_updater_config
            .build_critic_updater(&critic_variables);

        Self {
            observation_space,
            action_space,
            discount_factor,
            device: config.device,
            policy,
            policy_updater,
            policy_variables,
            cpu_policy,
            cpu_policy_variables,
            critic,
            critic_updater,
        }
    }
}

/*
impl<OS, AS, P, PU, C, CU> Actor<OS::Element, AS::Element>
    for ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: EncoderFeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: StatefulIterativeModule,
{
    fn act(&mut self, observation: &OS::Element) -> AS::Element {
        let _no_grad = tch::no_grad_guard();
        let observation_features = self.observation_space.features(observation);

        let policy = self.cpu_policy.as_mut().unwrap_or(&mut self.policy);
        let output = policy.step(&observation_features);
        self.action_space.sample_element(&output)
    }

    fn reset(&mut self) {
        self.cpu_policy.as_mut().unwrap_or(&mut self.policy).reset();
    }
}

impl<OS, AS, P, PU, C, CU> BatchUpdate<OS::Element, AS::Element>
    for ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: EncoderFeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: Policy,
    PU: UpdatePolicy<AS>,
    C: Critic,
    CU: UpdateCritic,
{
    fn batch_update(
        &mut self,
        history: &mut dyn HistoryBuffer<OS::Element, AS::Element>,
        logger: &mut dyn TimeSeriesLogger,
    ) {
        let features = LazyPackedHistoryFeatures::new(
            history.episodes(),
            &self.observation_space,
            &self.action_space,
            self.discount_factor,
            self.device,
        );

        let mut update_logger = logger.event_logger(Event::AgentOptPeriod);
        let mut history_logger = update_logger.scope("history");

        history_logger.unwrap_log_scalar("num_steps", features.num_steps() as f64);
        history_logger.unwrap_log_scalar("num_episodes", features.num_episodes() as f64);
        if features.is_empty() {
            history_logger.unwrap_log("no_model_update", "Skipping update; empty history");
            return;
        }

        let mut policy_logger = logger.scope("policy");
        let policy_stats = self.policy_updater.update_policy(
            &self.policy,
            &self.critic,
            &features,
            &self.action_space,
            &mut policy_logger,
        );
        if let Some(entropy) = policy_stats.entropy {
            policy_logger.unwrap_log_scalar(Event::AgentOptPeriod, "entropy", entropy);
        }

        let mut critic_logger = logger.scope("critic");
        self.critic_updater
            .update_critic(&self.critic, &features, &mut critic_logger);

        // Copy the updated variables to the CPU policy if there is one
        if let Some(ref mut cpu_policy_variables) = self.cpu_policy_variables {
            cpu_policy_variables
                .copy(&self.policy_variables)
                .expect("Variable mismatch between main policy and CPU policy");
        }

        logger.end_event(Event::AgentOptPeriod).unwrap();
    }
}
*/

impl<OS: Space, AS: Space, P, PU, C, CU> SetActorMode for ActorCriticAgent<OS, AS, P, PU, C, CU> {}
