use super::critics::StepValueTarget;
use super::features::{HistoryFeatures, LazyHistoryFeatures};
use super::schedules::{DataCollectionSchedule, ExplorationRateSchedule};
use super::{n_backward_steps, ToLog, WithCpuCopy};
use crate::agents::buffers::{HistoryDataBound, ReplayBuffer};
use crate::agents::{Actor, ActorMode, Agent, BatchUpdate, BuildAgent, BuildAgentError};
use crate::envs::EnvStructure;
use crate::feedback::Reward;
use crate::logging::StatsLogger;
use crate::spaces::{FeatureSpace, FiniteSpace, NonEmptyFeatures, ReprSpace, SampleSpace, Space};
use crate::torch::modules::{AsModule, BuildModule, Module, SeqIterative, SeqPacked};
use crate::torch::optimizers::{AdamConfig, BuildOptimizer, Optimizer};
use crate::torch::packed::PackedTensor;
use crate::torch::serialize::DeviceDef;
use crate::utils::sequence::Sequence;
use crate::Prng;
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::iter;
use std::rc::Rc;
use tch::{Device, Reduction, Tensor};

/// Configuration for [`DqnAgent`]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct DqnConfig<VB, OB = AdamConfig> {
    pub action_value_fn_config: VB,
    pub optimizer_config: OB,

    pub target: StepValueTarget,
    pub exploration_rate: ExplorationRateSchedule,
    pub minibatch_steps: usize,
    pub opt_steps_per_update: usize,
    pub buffer_capacity: usize,
    pub update_size: DataCollectionSchedule,

    #[serde(with = "DeviceDef")]
    pub device: Device,
}

impl<VB, OB> Default for DqnConfig<VB, OB>
where
    VB: Default,
    OB: Default,
{
    // The RainbowDQN paper uses the following values,
    // selected for single-thread training with large-observations:
    //
    // buffer_capacity: 1M
    // discount_factor 0.99
    // update_size (replay_period in paper): 4 steps
    // minibatch_size: 32 steps
    //
    // The issue is the time to sync updated parameters to the actors since the updates done on the
    // GPU while actions are done on the CPU (slow to move actor obs features onto gpu). Also
    // creating threads takes some time. The paper must be single-threaded.
    fn default() -> Self {
        Self {
            action_value_fn_config: VB::default(),
            optimizer_config: OB::default(),
            target: StepValueTarget::default(), // TODO: Default OneStepTD; might need double DQN
            exploration_rate: ExplorationRateSchedule::default(),
            minibatch_steps: 100_000,
            opt_steps_per_update: 50,
            buffer_capacity: 10_000_000,
            update_size: DataCollectionSchedule::FirstRest {
                first: 1_000_000,
                rest: 100_000,
            },
            device: Device::cuda_if_available(),
        }
    }
}

impl<OS, AS, FS, VB, OB> BuildAgent<OS, AS, FS> for DqnConfig<VB, OB>
where
    OS: FeatureSpace + Clone,
    OS::Element: 'static,
    AS: FiniteSpace + SampleSpace + ReprSpace<Tensor> + Clone,
    AS::Element: 'static,
    FS: Space<Element = Reward>,
    VB: BuildModule,
    VB::Module: SeqPacked + SeqIterative,
    OB: BuildOptimizer,
    OB::Optimizer: Optimizer,
{
    type Agent = DqnAgent<OS, AS, VB::Module, OB::Optimizer>;

    fn build_agent(
        &self,
        env: &dyn EnvStructure<ObservationSpace = OS, ActionSpace = AS, FeedbackSpace = FS>,
        rng: &mut Prng,
    ) -> Result<Self::Agent, BuildAgentError> {
        Ok(DqnAgent::new(env, self, Prng::from_rng(rng).unwrap()))
    }
}

/// Deep Q-Learning Agent
///
/// Based on
/// "[Playing Atari with Deep Reinforcement Learning][dqn]"
/// by Volodymyr Mnih et al. (2013)
/// and
/// "[Rainbow: Combining Improvements in Deep Reinforcement Learning][rainbow]"
/// by Hessel et al. (2017)
///
/// [dqn]: https://arxiv.org/pdf/1312.5602.pdf
/// [rainbow]: https://arxiv.org/pdf/1710.02298.pdf
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DqnAgent<OS, AS, V: AsModule, O> {
    observation_space: NonEmptyFeatures<OS>,
    action_space: AS,

    action_value_fn: WithCpuCopy<V>,
    optimizer: O,

    target: StepValueTarget,
    exploration_rate: ExplorationRateSchedule,
    minibatch_steps: usize,
    opt_steps_per_update: usize,
    /// Capacity of each individual buffer
    buffer_capacity: usize,
    update_size: DataCollectionSchedule,
    discount_factor: f32,

    /// Total number of collected steps in all updates.
    global_steps: u64,

    // Tensors will deserialize to CPU
    #[serde(skip, default = "cpu_device")]
    device: Device,

    /// Prngs for sampling batches in updates.
    rng: Prng,
}

impl<OS, AS, V, O> DqnAgent<OS, AS, V, O>
where
    OS: FeatureSpace,
    AS: FiniteSpace,
    V: AsModule,
{
    #[allow(clippy::cast_possible_truncation)]
    pub fn new<E, VB, OB>(env: &E, config: &DqnConfig<VB, OB>, rng: Prng) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        E::FeedbackSpace: Space<Element = Reward>,
        VB: BuildModule<Module = V>,
        OB: BuildOptimizer<Optimizer = O>,
    {
        let observation_space = NonEmptyFeatures::new(env.observation_space());
        let action_space = env.action_space();
        let num_observation_features = observation_space.num_features();
        let num_actions = action_space.size();

        let action_value_fn = config.action_value_fn_config.build_module(
            num_observation_features,
            num_actions,
            config.device,
        );

        let optimizer = config
            .optimizer_config
            .build_optimizer(action_value_fn.as_module().trainable_variables())
            .unwrap();

        Self {
            observation_space,
            action_space,
            action_value_fn: WithCpuCopy::new(action_value_fn, config.device),
            optimizer,
            target: config.target,
            exploration_rate: config.exploration_rate,
            minibatch_steps: config.minibatch_steps,
            opt_steps_per_update: config.opt_steps_per_update,
            buffer_capacity: config.buffer_capacity,
            update_size: config.update_size,
            discount_factor: env.discount_factor() as f32,
            global_steps: 0,
            device: config.device,
            rng,
        }
    }
}

const fn cpu_device() -> Device {
    Device::Cpu
}

impl<OS, AS, V, O> Agent<OS::Element, AS::Element> for DqnAgent<OS, AS, V, O>
where
    OS: FeatureSpace + Clone,
    OS::Element: 'static,
    AS: FiniteSpace + SampleSpace + ReprSpace<Tensor> + Clone,
    AS::Element: 'static,
    V: AsModule,
    V::Module: SeqPacked + SeqIterative,
    O: Optimizer,
{
    type Actor = DqnActor<OS, AS, V::Module>;

    fn actor(&self, mode: ActorMode) -> Self::Actor {
        let exploration_rate = self
            .exploration_rate
            .exploration_rate(self.global_steps, mode);
        DqnActor {
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
            action_value_fn: self.action_value_fn.shallow_clone_module_cpu(),
            exploration_rate,
        }
    }
}

impl<OS, AS, V, O> BatchUpdate<OS::Element, AS::Element> for DqnAgent<OS, AS, V, O>
where
    OS: FeatureSpace,
    OS::Element: 'static,
    AS: FiniteSpace + ReprSpace<Tensor>,
    AS::Element: 'static,
    V: AsModule,
    V::Module: SeqPacked,
    O: Optimizer,
{
    type Feedback = Reward;
    type HistoryBuffer = ReplayBuffer<OS::Element, AS::Element>;

    fn buffer(&self) -> Self::HistoryBuffer {
        ReplayBuffer::with_capacity(self.buffer_capacity)
    }

    fn min_update_size(&self) -> HistoryDataBound {
        self.update_size.update_size(self.global_steps)
    }

    fn batch_update<'a, I>(&mut self, buffers: I, logger: &mut dyn StatsLogger)
    where
        Self: Sized,
        I: IntoIterator<Item = &'a mut Self::HistoryBuffer>,
        Self::HistoryBuffer: 'a,
    {
        self.batch_update_slice_refs(&mut buffers.into_iter().collect::<Vec<_>>(), logger)
    }
}

impl<OS, AS, V, O> DqnAgent<OS, AS, V, O>
where
    OS: FeatureSpace,
    OS::Element: 'static,
    AS: FiniteSpace + ReprSpace<Tensor>,
    AS::Element: 'static,
    V: AsModule,
    V::Module: SeqPacked,
    O: Optimizer,
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
        buffers: &mut [&mut ReplayBuffer<OS::Element, AS::Element>],
        logger: &mut dyn StatsLogger,
    ) {
        logger.log_scalar(
            "exploration_rate",
            self.exploration_rate
                .exploration_rate(self.global_steps, ActorMode::Training),
        );

        // Update the global step count.
        self.global_steps = buffers.iter().map(|b| b.total_step_count()).sum();

        // Mutably borrow the action value fn to invalidate any CPU copy
        let _ = self.action_value_fn.as_module_mut();

        let sample_minibatch = || {
            let sampled_episodes = iter::repeat(&*buffers).flatten().map(|buf| {
                buf.episodes()
                    .get(Uniform::new(0usize, buf.num_episodes()).sample(&mut self.rng))
                    .unwrap()
            });
            let mut total_steps = 0;
            let minibatch_episodes = sampled_episodes.take_while(|ep| {
                let take = total_steps < self.minibatch_steps;
                total_steps += ep.len();
                take
            });
            let features = LazyHistoryFeatures::new(
                minibatch_episodes,
                &self.observation_space,
                &self.action_space,
                self.device,
            );

            // TODO: Separate policy/target networks (double DQN)
            let targets = tch::no_grad(|| {
                self.target.targets(
                    &self
                        .action_value_fn
                        .as_module()
                        .batch_map(|action_values| action_values.amax(&[-1], false)),
                    self.discount_factor,
                    &features,
                )
            });
            let observations = features.observation_features();
            let actions = features.actions().tensor().unsqueeze(-1);

            Rc::new((observations.clone(), actions, targets))
        };

        let loss_fn = |data: Rc<(PackedTensor, Tensor, PackedTensor)>| {
            let (observations, actions, targets) = data.as_ref();
            let action_values = self
                .action_value_fn
                .as_module()
                .seq_packed(observations)
                .tensor()
                .gather(-1, actions, false)
                .squeeze_dim(-1);
            action_values.mse_loss(targets.tensor(), Reduction::Mean)
        };

        n_backward_steps(
            &mut self.optimizer,
            sample_minibatch,
            loss_fn,
            self.opt_steps_per_update as u64,
            logger,
            ToLog::All,
            "action value update error",
        );
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct DqnActor<OS, AS, V> {
    observation_space: NonEmptyFeatures<OS>,
    action_space: AS,
    action_value_fn: V,
    exploration_rate: f64,
}

impl<OS, AS, V> Actor<OS::Element, AS::Element> for DqnActor<OS, AS, V>
where
    OS: FeatureSpace,
    AS: FiniteSpace + SampleSpace,
    V: SeqIterative,
{
    type EpisodeState = V::State;

    fn initial_state(&self, _: &mut Prng) -> Self::EpisodeState {
        self.action_value_fn.initial_state()
    }

    fn act(
        &self,
        episode_state: &mut Self::EpisodeState,
        observation: &OS::Element,
        rng: &mut Prng,
    ) -> AS::Element {
        if rng.gen_bool(self.exploration_rate) {
            return self.action_space.sample(rng);
        }

        let _no_grad = tch::no_grad_guard();
        let observation_features: Tensor = self.observation_space.features(observation);
        let action_values = self
            .action_value_fn
            .step(episode_state, &observation_features);
        let action_index: i64 = action_values.argmax(None, false).into();
        self.action_space
            .from_index(action_index.try_into().unwrap())
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::super::critics::StepValueTarget;
    use super::*;
    use crate::agents::testing;
    use crate::torch::modules::{BuildModule, GruMlpConfig, MlpConfig, SeqIterative, SeqPacked};
    use crate::torch::optimizers::AdamConfig;
    use rstest::rstest;

    #[rstest]
    fn learns_deterministic_bandit<MB>(
        #[values(MlpConfig::default(), GruMlpConfig::default())] module: MB,
        #[values(StepValueTarget::RewardToGo, StepValueTarget::OneStepTd)] target: StepValueTarget,
        #[values(Device::Cpu, Device::cuda_if_available())] device: Device,
    ) where
        MB: BuildModule + Default,
        MB::Module: SeqPacked + SeqIterative,
    {
        let config = DqnConfig {
            action_value_fn_config: module,
            optimizer_config: AdamConfig {
                learning_rate: 0.1,
                ..AdamConfig::default()
            },
            target,
            minibatch_steps: 4,
            buffer_capacity: 20,
            update_size: DataCollectionSchedule::FirstRest { first: 10, rest: 4 },
            device,
            ..Default::default()
        };
        testing::train_deterministic_bandit(&config, 10, 0.9);
    }
}
