//! Actor-critic agent
use super::{
    actor::PolicyActor,
    critic::Critic,
    features::LazyPackedHistoryFeatures,
    learning_critic::{BuildLearningCritic, LearningCritic},
    learning_policy::{BuildLearningPolicy, LearningPolicy},
};
use crate::{
    agents::buffers::{BufferCapacityBound, SimpleBuffer, WriteHistoryBuffer},
    agents::{ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError},
    envs::EnvStructure,
    logging::StatsLogger,
    spaces::{FeatureSpace, NonEmptyFeatures, ParameterizedDistributionSpace},
    torch::modules::Module,
    utils::torch::DeviceDef,
    Prng,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::time::Instant;
use tch::{Device, Tensor};

/// Configuration for [`ActorCriticAgent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorCriticConfig<PB, CB> {
    pub policy_config: PB,
    pub critic_config: CB,
    /// Minimum number of collected steps per batch update
    pub min_batch_steps: usize,
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
            min_batch_steps: 10_000,
            device: Device::cuda_if_available(),
        }
    }
}

impl<OS, AS, PB, CB> BuildAgent<OS, AS> for ActorCriticConfig<PB, CB>
where
    OS: FeatureSpace + Clone,
    OS::Element: 'static,
    AS: ParameterizedDistributionSpace<Tensor> + Clone,
    AS::Element: 'static,
    PB: BuildLearningPolicy,
    CB: BuildLearningCritic,
{
    type Agent = ActorCriticAgent<OS, AS, PB::LearningPolicy, CB::LearningCritic>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(ActorCriticAgent::new(env, self))
    }
}

/// Actor-crtic agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActorCriticAgent<OS, AS, P, C>
where
    P: LearningPolicy,
{
    observation_space: NonEmptyFeatures<OS>,
    action_space: AS,
    discount_factor: f64,
    min_batch_steps: usize,

    learning_policy: P,
    learning_critic: C,

    /// A copy of the policy stored on the CPU device for memory sharing with actors.
    ///
    /// Only created if `device` is not `Device::Cpu`.
    /// Must be re-copied after each policy update.
    #[serde(skip, default)]
    cpu_policy: RefCell<Option<P::Policy>>,

    // Tensors will deserialize to CPU
    #[serde(skip, default = "cpu_device")]
    device: Device,
}

const fn cpu_device() -> Device {
    Device::Cpu
}

impl<OS, AS, P, C> ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: LearningPolicy,
    C: LearningCritic,
{
    pub fn new<E, PB, CB>(env: &E, config: &ActorCriticConfig<PB, CB>) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        PB: BuildLearningPolicy<LearningPolicy = P>,
        CB: BuildLearningCritic<LearningCritic = C>,
    {
        let observation_space = NonEmptyFeatures::new(env.observation_space());
        let action_space = env.action_space();
        let num_observation_features = observation_space.num_features();

        let learning_policy = config.policy_config.build_learning_policy(
            num_observation_features,
            action_space.num_distribution_params(),
            config.device,
        );

        let learning_critic = config
            .critic_config
            .build_learning_critic(num_observation_features, config.device);

        Self {
            observation_space,
            action_space,
            discount_factor: learning_critic
                .critic_ref()
                .discount_factor(env.discount_factor()),
            min_batch_steps: config.min_batch_steps,
            learning_policy,
            learning_critic,
            cpu_policy: RefCell::new(None),
            device: config.device,
        }
    }
}

impl<OS, AS, P, C> Agent<OS::Element, AS::Element> for ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace + Clone,
    OS::Element: 'static,
    AS: ParameterizedDistributionSpace<Tensor> + Clone,
    AS::Element: 'static,
    P: LearningPolicy,
    C: LearningCritic,
{
    type Actor = PolicyActor<OS, AS, P::Policy>;

    fn actor(&self, _: ActorMode) -> Self::Actor {
        let cpu_policy = if matches!(self.device, Device::Cpu) {
            self.learning_policy.policy_ref().shallow_clone()
        } else {
            self.cpu_policy
                .borrow_mut()
                .get_or_insert_with(|| {
                    self.learning_policy
                        .policy_ref()
                        .clone_to_device(Device::Cpu)
                })
                .shallow_clone()
        };

        PolicyActor::new(
            self.observation_space.clone(),
            self.action_space.clone(),
            cpu_policy,
        )
    }
}

impl<OS, AS, P, C> BatchUpdate<OS::Element, AS::Element> for ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace,
    OS::Element: 'static,
    AS: ParameterizedDistributionSpace<Tensor>,
    AS::Element: 'static,
    P: LearningPolicy,
    C: LearningCritic,
{
    type HistoryBuffer = SimpleBuffer<OS::Element, AS::Element>;

    fn batch_size_hint(&self) -> BufferCapacityBound {
        BufferCapacityBound {
            min_steps: self.min_batch_steps,
            min_episodes: 0,
            min_incomplete_episode_len: Some(100),
        }
    }

    fn buffer(&self, capacity: BufferCapacityBound) -> Self::HistoryBuffer {
        SimpleBuffer::new(capacity)
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn StatsLogger)
    where
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        self.batch_update_slice_refs(&mut buffers.into_iter().collect::<Vec<_>>(), logger);
    }

    fn batch_update_single(
        &mut self,
        buffer: &mut Self::HistoryBuffer,
        logger: &mut dyn StatsLogger,
    ) {
        self.batch_update_slice_refs(&mut [buffer], logger)
    }

    fn batch_update_slice(
        &mut self,
        buffers: &mut [Self::HistoryBuffer],
        logger: &mut dyn StatsLogger,
    ) {
        self.batch_update(buffers, logger)
    }
}

impl<OS, AS, P, C> ActorCriticAgent<OS, AS, P, C>
where
    OS: FeatureSpace,
    OS::Element: 'static,
    AS: ParameterizedDistributionSpace<Tensor>,
    AS::Element: 'static,
    P: LearningPolicy,
    C: LearningCritic,
{
    // Takes a slice of references because
    // * it iterates over the buffers twice and it is awkward to make the right bounds for
    //      a "clone-able" (actually, into_iter with shorter lifetimes) generic iterator.
    // * the function is relatively large and this avoids duplicate monomorphizations
    // * any inefficiency in the buffer access should be insignificant compared to the runtime
    //      cost of the rest of the update
    /// Batch update given a slice of buffer references
    fn batch_update_slice_refs(
        &mut self,
        buffers: &mut [&mut SimpleBuffer<OS::Element, AS::Element>],
        logger: &mut dyn StatsLogger,
    ) {
        let agent_update_start = Instant::now();
        // About to update the policy so clear any existing CPU policy copy
        self.cpu_policy = RefCell::new(None);

        let features = LazyPackedHistoryFeatures::new(
            buffers.iter().flat_map(|b| b.episodes()),
            &self.observation_space,
            &self.action_space,
            self.discount_factor,
            self.device,
        );
        let mut history_logger = logger.with_scope("history");
        let num_steps = features.num_steps();
        let num_episodes = features.num_episodes();
        history_logger.log_scalar("num_steps", num_steps as f64);
        history_logger.log_scalar("num_episodes", num_episodes as f64);
        history_logger.log_counter_increment("cumulative_steps", num_steps as u64);
        history_logger.log_counter_increment("cumulative_episodes", num_episodes as u64);
        if features.is_empty() {
            history_logger.log_message("no_model_update", "Skipping update; empty history");
            return;
        }

        let mut policy_logger = logger.with_scope("policy");
        let policy_update_start = Instant::now();
        let policy_stats = self.learning_policy.update_policy(
            self.learning_critic.critic_ref(),
            &features,
            &self.action_space,
            &mut policy_logger,
        );
        policy_logger.log_duration("update_time", policy_update_start.elapsed());
        if let Some(entropy) = policy_stats.entropy {
            policy_logger.log_scalar("entropy", entropy);
        }

        let mut critic_logger = logger.with_scope("critic");
        let critic_update_start = Instant::now();
        self.learning_critic
            .update_critic(&features, &mut critic_logger);
        critic_logger.log_duration("update_time", critic_update_start.elapsed());

        for buffer in buffers {
            buffer.clear()
        }

        let mut agent_logger = logger.with_scope("agent");
        agent_logger.log_duration("update_time", agent_update_start.elapsed());
        agent_logger.log_counter_increment("update_count", 1)
    }
}
