//! Basic policy network actor
use super::super::history::{HistoryBuffer, LazyPackedHistoryFeatures};
use super::super::seq_modules::StatefulIterativeModule;
use super::super::step_value::{StepValue, StepValueBuilder};
use super::super::ModuleBuilder;
use crate::logging::{Event, TimeSeriesLogger};
use crate::spaces::{
    BaseFeatureSpace, FeatureSpace, NonEmptyFeatures, ParameterizedDistributionSpace, ReprSpace,
    Space,
};
use crate::{Actor, EnvStructure, Step};
use std::num::NonZeroUsize;
use tch::{nn::VarStore, Device, Tensor};

/// Configuration for [`PolicyValueNetActor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolicyValueNetActorConfig<PB, VB> {
    pub steps_per_epoch: usize,
    pub include_incomplete_episode_len: Option<NonZeroUsize>,
    pub value_train_iters: u64,
    pub policy_config: PB,
    pub value_config: VB,
    pub device: Device,
}

impl<PB, VB> PolicyValueNetActorConfig<PB, VB> {
    pub const fn new(
        steps_per_epoch: usize,
        include_incomplete_episode_len: Option<NonZeroUsize>,
        value_train_iters: u64,
        policy_config: PB,
        value_config: VB,
        device: Device,
    ) -> Self {
        Self {
            steps_per_epoch,
            include_incomplete_episode_len,
            value_train_iters,
            policy_config,
            value_config,
            device,
        }
    }
}

impl<PB, VB> Default for PolicyValueNetActorConfig<PB, VB>
where
    PB: Default,
    VB: Default,
{
    fn default() -> Self {
        Self {
            steps_per_epoch: 1000,
            include_incomplete_episode_len: Some(NonZeroUsize::new(10).unwrap()),
            value_train_iters: 80,
            policy_config: Default::default(),
            value_config: Default::default(),
            device: Device::Cpu,
        }
    }
}

impl<PB, VB> PolicyValueNetActorConfig<PB, VB> {
    pub fn build_actor<E, P, V>(
        &self,
        env: &E,
    ) -> PolicyValueNetActor<E::ObservationSpace, E::ActionSpace, P, V>
    where
        E: EnvStructure + ?Sized,
        <E as EnvStructure>::ObservationSpace: Space + BaseFeatureSpace,
        <E as EnvStructure>::ActionSpace: ParameterizedDistributionSpace<Tensor>,
        PB: ModuleBuilder<P>,
        VB: StepValueBuilder<V>,
        V: StepValue,
    {
        PolicyValueNetActor::new(env, self)
    }
}

/// Policy and Value Network Actor
///
/// Actor that maintains a policy and a value network.
/// Takes actions and records history.
///
/// Accepts a callback function to update the model parameters.
#[derive(Debug)]
pub struct PolicyValueNetActor<OS, AS, P, V>
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

    /// Number of step value update iterations per epoch.
    pub value_train_iters: u64,

    /// Device on which model variables are stored.
    pub device: Device,

    /// The policy module.
    pub policy: P,

    /// Policy module variables.
    pub policy_variables: VarStore,

    /// A copy of the policy module (including parameters) on the CPU for fast actions.
    ///
    /// This is defined if the main policy module is not on the CPU.
    cpu_policy: Option<P>,

    /// Cpu policy variables if the main policy is not on the CPU.
    cpu_policy_variables: Option<VarStore>,

    /// The value estimator module.
    pub value: V,

    /// Value estimator module variables
    pub value_variables: VarStore,

    /// The recorded step history.
    history: HistoryBuffer<OS::Element, AS::Element>,
}

impl<OS, AS, P, V> PolicyValueNetActor<OS, AS, P, V>
where
    OS: Space + BaseFeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    V: StepValue,
{
    /// Create a new `PolicyValueNetActor`
    ///
    /// # Args
    /// * `env` - Environment structure.
    /// * `config` - `PolicyValueNetActor` configuration parameters.
    /// * `device` - Device on which the policy and value networks are stored.
    ///              If this is not CPU, a copy of the policy is maintained on the CPU for faster
    ///              actions.
    pub fn new<E, PB, VB>(env: &E, config: &PolicyValueNetActorConfig<PB, VB>) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        PB: ModuleBuilder<P>,
        VB: StepValueBuilder<V>,
    {
        let observation_space = NonEmptyFeatures::new(env.observation_space());
        let action_space = env.action_space();
        let max_steps_per_epoch = config.steps_per_epoch + config.steps_per_epoch / 10;

        let policy_variables = VarStore::new(config.device);
        let policy = config.policy_config.build_module(
            &policy_variables.root(),
            observation_space.num_features(),
            action_space.num_distribution_params(),
        );

        let (cpu_policy, cpu_policy_variables) = if config.device != Device::Cpu {
            let mut cpu_policy_variables = VarStore::new(Device::Cpu);
            let cpu_policy = config.policy_config.build_module(
                &cpu_policy_variables.root(),
                observation_space.num_features(),
                action_space.num_distribution_params(),
            );
            cpu_policy_variables.copy(&policy_variables).unwrap();
            (Some(cpu_policy), Some(cpu_policy_variables))
        } else {
            (None, None)
        };

        let value_variables = VarStore::new(config.device);
        let value = config
            .value_config
            .build_step_value(&value_variables.root(), observation_space.num_features());
        let discount_factor = value.discount_factor(env.discount_factor());

        Self {
            observation_space,
            action_space,
            discount_factor,
            steps_per_epoch: config.steps_per_epoch,
            max_steps_per_epoch,
            value_train_iters: config.value_train_iters,
            device: config.device,
            policy,
            policy_variables,
            cpu_policy,
            cpu_policy_variables,
            value,
            value_variables,
            history: HistoryBuffer::new(
                Some(max_steps_per_epoch),
                config.include_incomplete_episode_len,
            ),
        }
    }
}

impl<OS, AS, P, V> Actor<OS::Element, AS::Element> for PolicyValueNetActor<OS, AS, P, V>
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

/// History features type for [`PolicyValueNetActor`]
pub type HistoryFeatures<'a, OS, AS> = LazyPackedHistoryFeatures<'a, NonEmptyFeatures<OS>, AS>;

impl<OS, AS, P, V> PolicyValueNetActor<OS, AS, P, V>
where
    OS: FeatureSpace<Tensor>,
    AS: ReprSpace<Tensor>,
    V: StepValue,
{
    /// Perform a step update and update the model paramters if enough data has been accumulated.
    ///
    /// # Args
    ///
    /// * `step` - The most recent environment step.
    ///
    /// * `update_policy` - Callback function that performs a policy update
    ///     and returns an estimate of the policy entropy.
    ///
    /// * `update_value` - Callback function that performs a value net update
    ///     and returns the value function loss (pre- or post-update).
    ///     Will be called `value_train_iters` times if `value.trainable()` is true.
    ///
    /// * `logger` - Logger to which epoch statistics are logged. Forwarded to the callbacks.

    pub fn update<F, G>(
        &mut self,
        step: Step<OS::Element, AS::Element>,
        update_policy: F,
        update_value: G,
        logger: &mut dyn TimeSeriesLogger,
    ) where
        F: FnOnce(&Self, &HistoryFeatures<OS, AS>, &mut dyn TimeSeriesLogger) -> Option<Tensor>,
        G: FnMut(&Self, &HistoryFeatures<OS, AS>, &mut dyn TimeSeriesLogger) -> Option<Tensor>,
    {
        let episode_done = step.episode_done;
        self.history.push(step);

        let history_len = self.history.len();
        if history_len >= self.max_steps_per_epoch
            || (history_len >= self.steps_per_epoch && episode_done)
        {
            self.epoch_update(update_policy, update_value, logger);
        }
    }

    /// Update the model parameters and clear the stored history.
    ///
    /// * `update_policy` - Callback function that performs a policy update
    ///     and optionally returns an estimate of the policy entropy.
    ///
    /// * `update_value` - Callback function that performs a value net update
    ///     and optionally returns the value function loss (pre- or post-update).
    ///     Will be called `value_train_iters` times if `value.trainable()` is true.
    ///
    /// * `logger` - Logger to which epoch statistics are logged. Forwarded to the callbacks.

    fn epoch_update<F, G>(
        &mut self,
        update_policy: F,
        mut update_value: G,
        logger: &mut dyn TimeSeriesLogger,
    ) where
        F: FnOnce(&Self, &HistoryFeatures<OS, AS>, &mut dyn TimeSeriesLogger) -> Option<Tensor>,
        G: FnMut(&Self, &HistoryFeatures<OS, AS>, &mut dyn TimeSeriesLogger) -> Option<Tensor>,
    {
        let features = LazyPackedHistoryFeatures::new(
            self.history.steps(),
            self.history.episode_ranges(),
            &self.observation_space,
            &self.action_space,
            self.discount_factor,
            self.device,
        );
        epoch_log_scalar(logger, "num_steps", self.history.len() as f64);
        epoch_log_scalar(logger, "num_episodes", self.history.num_episodes() as f64);

        if let Some(entropy) = update_policy(self, &features, logger) {
            epoch_log_scalar(logger, "policy_entropy", &entropy);
        }

        if self.value.trainable() {
            for i in 0..self.value_train_iters {
                let value_loss = update_value(self, &features, logger);

                if let Some(value_loss) = value_loss {
                    if i == 0 {
                        epoch_log_scalar(logger, "value_loss_initial", &value_loss);
                    } else if i == self.value_train_iters - 1 {
                        epoch_log_scalar(logger, "value_loss_final", &value_loss);
                    }
                }
            }
        }

        // Copy the updated variables to the CPU policy if there is one
        if let Some(ref mut cpu_policy_variables) = self.cpu_policy_variables {
            cpu_policy_variables
                .copy(&self.policy_variables)
                .expect("Variable mismatch between main policy and CPU policy");
        }

        logger.end_event(Event::Epoch);

        self.history.clear();
    }
}

/// Log a value with the epoch event.
fn epoch_log_scalar<L, V>(logger: &mut L, name: &str, value: V)
where
    L: TimeSeriesLogger + ?Sized,
    V: Into<f64>,
{
    logger.log(Event::Epoch, name, value.into().into()).unwrap();
}
