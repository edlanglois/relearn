use super::super::{
    critic::{BuildCritic, Critic},
    features::LazyPackedHistoryFeatures,
    modules::{BuildModule, IterativeModule, Module, SequenceModule},
    updaters::{BuildCriticUpdater, BuildPolicyUpdater, UpdateCritic, UpdatePolicy},
};
use crate::agents::buffers::{BufferCapacityBound, SimpleBuffer, WriteHistoryBuffer};
use crate::agents::{Actor, ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError};
use crate::envs::EnvStructure;
use crate::logging::StatsLogger;
use crate::spaces::{FeatureSpace, NonEmptyFeatures, ParameterizedDistributionSpace, ReprSpace};
use crate::utils::torch::DeviceDef;
use crate::Prng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tch::{Device, Tensor};

/// Configuration for [`ActorCriticAgent`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorCriticConfig<PB, PUB, CB, CUB> {
    pub policy_config: PB,
    pub policy_updater_config: PUB,
    pub critic_config: CB,
    pub critic_updater_config: CUB,
    pub min_batch_steps: usize,
    #[serde(with = "DeviceDef")]
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
            min_batch_steps: 10_000,
            device: Device::Cpu,
        }
    }
}

impl<PB, PUB, CB, CUB, OS, AS> BuildAgent<OS, AS> for ActorCriticConfig<PB, PUB, CB, CUB>
where
    OS: FeatureSpace + 'static,
    AS: ParameterizedDistributionSpace<Tensor> + 'static,
    PB: BuildModule + Clone,
    PB::Module: SequenceModule + IterativeModule,
    PUB: BuildPolicyUpdater<AS>,
    PUB::Updater: UpdatePolicy<AS>,
    CB: BuildCritic,
    CB::Critic: Critic,
    CUB: BuildCriticUpdater,
    CUB::Updater: UpdateCritic,
{
    #[allow(clippy::type_complexity)]
    type Agent = ActorCriticAgent<OS, AS, PB::Module, PUB::Updater, CB::Critic, CUB::Updater>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS>,
        _: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(ActorCriticAgent::new(env, self))
    }
}

/// Actor-critic agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActorCriticAgent<OS, AS, P, PU, C, CU> {
    spaces: Arc<Spaces<OS, AS>>,
    min_batch_steps: usize,
    discount_factor: f64,

    policy: P,
    policy_updater: PU,

    critic: C,
    critic_updater: CU,

    // Tensors will deserialize to CPU
    #[serde(skip, default = "cpu_device")]
    device: Device,
}

const fn cpu_device() -> Device {
    Device::Cpu
}

impl<OS, AS, P, PU, C, CU> ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: FeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: Module,
    C: Critic,
{
    pub fn new<E, PB, PUB, CB, CUB>(env: &E, config: &ActorCriticConfig<PB, PUB, CB, CUB>) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        PB: BuildModule<Module = P> + Clone,
        PUB: BuildPolicyUpdater<AS, Updater = PU>,
        CB: BuildCritic<Critic = C>,
        CUB: BuildCriticUpdater<Updater = CU>,
    {
        let observation_space = NonEmptyFeatures::new(env.observation_space());
        let action_space = env.action_space();

        let policy = config.policy_config.build_module(
            observation_space.num_features(),
            action_space.num_distribution_params(),
            config.device,
        );
        let policy_updater = config
            .policy_updater_config
            .build_policy_updater(policy.trainable_variables());

        let critic = config
            .critic_config
            .build_critic(observation_space.num_features(), config.device);
        let discount_factor = critic.discount_factor(env.discount_factor());
        let critic_updater = config
            .critic_updater_config
            .build_critic_updater(critic.trainable_variables());

        Self {
            spaces: Arc::new(Spaces {
                observation_space,
                action_space,
            }),
            min_batch_steps: config.min_batch_steps,
            discount_factor,
            policy,
            policy_updater,
            critic,
            critic_updater,
            device: config.device,
        }
    }
}

impl<OS, AS, P, PU, C, CU> Agent<OS::Element, AS::Element>
    for ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: FeatureSpace + 'static,
    AS: ParameterizedDistributionSpace<Tensor> + 'static,
    P: SequenceModule + IterativeModule,
    PU: UpdatePolicy<AS>,
    C: Critic,
    CU: UpdateCritic,
{
    type Actor = ActorCriticActor<OS, AS, P>;

    fn actor(&self, _: ActorMode) -> Self::Actor {
        // TODO: Store cpu_policy in the agent and synchronize it to `policy` after every update.
        // Then use shallow_clone() to produce actor policies.
        ActorCriticActor::new(
            Arc::clone(&self.spaces),
            self.policy.clone_to_device(Device::Cpu),
        )
    }
}

impl<OS, AS, P, PU, C, CU> BatchUpdate<OS::Element, AS::Element>
    for ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: FeatureSpace + 'static,
    AS: ReprSpace<Tensor> + 'static,
    P: SequenceModule,
    PU: UpdatePolicy<AS>,
    C: Critic,
    CU: UpdateCritic,
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

impl<OS, AS, P, PU, C, CU> ActorCriticAgent<OS, AS, P, PU, C, CU>
where
    OS: FeatureSpace + 'static,
    AS: ReprSpace<Tensor> + 'static,
    P: SequenceModule,
    PU: UpdatePolicy<AS>,
    C: Critic,
    CU: UpdateCritic,
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

        let features = LazyPackedHistoryFeatures::new(
            buffers.iter().flat_map(|b| b.episodes()),
            &self.spaces.observation_space,
            &self.spaces.action_space,
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
        let policy_stats = self.policy_updater.update_policy(
            &self.policy,
            &self.critic,
            &features,
            &self.spaces.action_space,
            &mut policy_logger,
        );
        policy_logger.log_duration("update_time", policy_update_start.elapsed());
        if let Some(entropy) = policy_stats.entropy {
            policy_logger.log_scalar("entropy", entropy);
        }

        let mut critic_logger = logger.with_scope("critic");
        let critic_update_start = Instant::now();
        self.critic_updater
            .update_critic(&self.critic, &features, &mut critic_logger);
        critic_logger.log_duration("update_time", critic_update_start.elapsed());

        for buffer in buffers {
            buffer.clear()
        }

        logger.log_duration("agent/update_time", agent_update_start.elapsed());
        logger.log_counter_increment("agent/update_count", 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorCriticActor<OS, AS, P> {
    spaces: Arc<Spaces<OS, AS>>,
    policy: P,
}

impl<OS: FeatureSpace, AS, P> ActorCriticActor<OS, AS, P> {
    pub fn new(spaces: Arc<Spaces<OS, AS>>, policy: P) -> Self {
        Self { spaces, policy }
    }
}

impl<OS, AS, P> Actor<OS::Element, AS::Element> for ActorCriticActor<OS, AS, P>
where
    OS: FeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: IterativeModule,
{
    type EpisodeState = P::State;

    fn new_episode_state(&self, _: &mut Prng) -> Self::EpisodeState {
        self.policy.initial_state()
    }

    fn act(
        &self,
        state: &mut Self::EpisodeState,
        observation: &OS::Element,
        _: &mut Prng,
    ) -> AS::Element {
        let _no_grad = tch::no_grad_guard();
        let input = self
            .spaces
            .observation_space
            .features::<Tensor>(observation);
        let output = self.policy.step(state, &input);
        self.spaces.action_space.sample_element(&output)
    }
}

/// Action and Observation spaces for `ActorCriticAgent`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Spaces<OS, AS> {
    observation_space: NonEmptyFeatures<OS>,
    action_space: AS,
}
