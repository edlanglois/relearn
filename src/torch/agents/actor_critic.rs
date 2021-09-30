use super::super::{
    critic::{BuildCritic, Critic},
    history::{HistoryBuffer, LazyPackedHistoryFeatures},
    policy::{BuildPolicy, Policy},
    seq_modules::StatefulIterativeModule,
    updaters::{BuildCriticUpdater, BuildPolicyUpdater, UpdateCritic, UpdatePolicy},
};
use crate::agents::{Actor, Agent, BuildAgent, BuildAgentError, SetActorMode, Step};
use crate::envs::EnvStructure;
use crate::logging::{Event, Logger, LoggerHelper, TimeSeriesLogger, TimeSeriesLoggerHelper};
use crate::spaces::{
    BaseFeatureSpace, BatchFeatureSpace, FeatureSpace, NonEmptyFeatures,
    ParameterizedDistributionSpace, ReprSpace, Space,
};
use std::num::NonZeroUsize;
use tch::{nn::VarStore, Device, Tensor};

/// Configuration for [`ActorCriticAgent`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActorCriticConfig<PB, PUB, CB, CUB> {
    pub steps_per_epoch: usize,
    pub include_incomplete_episode_len: Option<NonZeroUsize>,
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
            steps_per_epoch: 1000,
            include_incomplete_episode_len: Some(NonZeroUsize::new(10).unwrap()),
            policy_config: Default::default(),
            policy_updater_config: Default::default(),
            critic_config: Default::default(),
            critic_updater_config: Default::default(),
            device: Device::Cpu,
        }
    }
}

impl<PB, PUB, CB, CUB, E> BuildAgent<E> for ActorCriticConfig<PB, PUB, CB, CUB>
where
    PB: BuildPolicy,
    PUB: BuildPolicyUpdater<<E as EnvStructure>::ActionSpace>,
    CB: BuildCritic,
    CUB: BuildCriticUpdater,
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: FeatureSpace<Tensor> + BatchFeatureSpace<Tensor>,
    <E as EnvStructure>::ActionSpace: ReprSpace<Tensor> + ParameterizedDistributionSpace<Tensor>,
{
    type Agent = ActorCriticAgent<
        E::ObservationSpace,
        E::ActionSpace,
        PB::Policy,
        PUB::Updater,
        CB::Critic,
        CUB::Updater,
    >;

    fn build_agent(&self, env: &E, _seed: u64) -> Result<Self::Agent, BuildAgentError> {
        Ok(ActorCriticAgent::new(env, self))
    }
}

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

    /// Minimum number of steps to collect per epoch.
    ///
    /// This value is exceeded by search for the next episode boundary,
    /// up to a maximum of `max_steps_per_epoch`.
    pub steps_per_epoch: usize,

    /// Maximum number of steps per epoch.
    ///
    /// The actual number of steps is the first episode end
    /// between `steps_per_epoch` and `max_steps_per_epoch`,
    /// or `max_steps_per_epoch` if no episode end is found.
    pub max_steps_per_epoch: usize,

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

    /// The recorded step history.
    history: HistoryBuffer<OS::Element, AS::Element>,
}

impl<OS, AS, P, PU, C, CU> ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: Space + BaseFeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
{
    fn new<E, PB, PUB, CB, CUB>(env: &E, config: &ActorCriticConfig<PB, PUB, CB, CUB>) -> Self
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
        let max_steps_per_epoch = config.steps_per_epoch + config.steps_per_epoch / 10;

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

        let history = HistoryBuffer::new(
            Some(max_steps_per_epoch),
            config.include_incomplete_episode_len,
        );
        Self {
            observation_space,
            action_space,
            discount_factor,
            steps_per_epoch: config.steps_per_epoch,
            max_steps_per_epoch,
            device: config.device,
            policy,
            policy_updater,
            policy_variables,
            cpu_policy,
            cpu_policy_variables,
            critic,
            critic_updater,
            history,
        }
    }
}

impl<OS, AS, P, PU, C, CU> Actor<OS::Element, AS::Element>
    for ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: StatefulIterativeModule,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        let _no_grad = tch::no_grad_guard();
        let observation_features = self.observation_space.features(observation);

        let policy = self.cpu_policy.as_mut().unwrap_or(&mut self.policy);
        if new_episode {
            policy.reset();
        }

        let output = policy.step(&observation_features);
        self.action_space.sample_element(&output)
    }
}

impl<OS, AS, P, PU, C, CU> Agent<OS::Element, AS::Element>
    for ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: FeatureSpace<Tensor> + BatchFeatureSpace<Tensor>,
    AS: ReprSpace<Tensor> + ParameterizedDistributionSpace<Tensor>,
    P: Policy,
    PU: UpdatePolicy<AS>,
    C: Critic,
    CU: UpdateCritic,
{
    fn update(&mut self, step: Step<OS::Element, AS::Element>, logger: &mut dyn TimeSeriesLogger) {
        let episode_done = step.episode_done;
        self.history.push(step);

        let history_len = self.history.len();
        if history_len >= self.max_steps_per_epoch
            || (history_len >= self.steps_per_epoch && episode_done)
        {
            self.model_update(logger);
        }
    }
}

impl<OS, AS, P, PU, C, CU> ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: BatchFeatureSpace<Tensor>,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: Policy,
    PU: UpdatePolicy<AS>,
    C: Critic,
    CU: UpdateCritic,
{
    /// Update the actor and critic using the stored history buffer.
    ///
    /// Clears the history afterwards.
    fn model_update(&mut self, logger: &mut dyn TimeSeriesLogger) {
        let features = LazyPackedHistoryFeatures::new(
            self.history.steps(),
            self.history.episode_ranges(),
            &self.observation_space,
            &self.action_space,
            self.discount_factor,
            self.device,
        );
        let mut update_logger = logger.event_logger(Event::AgentOptPeriod);
        let mut history_logger = update_logger.scope("history");
        let episode_ranges = self.history.episode_ranges();
        let num_steps = episode_ranges.num_steps();

        history_logger.unwrap_log_scalar("num_steps", num_steps as f64);
        history_logger.unwrap_log_scalar("num_episodes", episode_ranges.len() as f64);
        if num_steps == 0 {
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

        self.history.clear();
        logger.end_event(Event::AgentOptPeriod).unwrap();
    }
}

impl<OS: Space, AS: Space, P, PU, C, CU> SetActorMode for ActorCriticAgent<OS, AS, P, PU, C, CU> {}
