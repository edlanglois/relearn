//! Basic policy network actor
use super::super::history::{HistoryBuffer, LazyPackedHistoryFeatures};
use super::super::seq_modules::StatefulIterativeModule;
use super::super::step_value::{StepValue, StepValueBuilder};
use super::super::ModuleBuilder;
use crate::logging::{Event, Logger};
use crate::spaces::{FeatureSpace, ParameterizedDistributionSpace, Space};
use crate::{Actor, EnvStructure, Step};
use tch::{nn::Path, Tensor};

/// Configuration for [`PolicyValueNetActor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyValueNetActorConfig<PB, VB> {
    pub steps_per_epoch: usize,
    pub value_train_iters: u64,
    pub policy_config: PB,
    pub value_config: VB,
}

impl<PB, VB> PolicyValueNetActorConfig<PB, VB> {
    pub const fn new(
        steps_per_epoch: usize,
        value_train_iters: u64,
        policy_config: PB,
        value_config: VB,
    ) -> Self {
        Self {
            steps_per_epoch,
            value_train_iters,
            policy_config,
            value_config,
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
            value_train_iters: 80,
            policy_config: Default::default(),
            value_config: Default::default(),
        }
    }
}

impl<PB, VB> PolicyValueNetActorConfig<PB, VB> {
    pub fn build_actor<OS, AS, P, V>(
        &self,
        env: EnvStructure<OS, AS>,
        policy_vs: &Path,
        value_vs: &Path,
    ) -> PolicyValueNetActor<OS, AS, P, V>
    where
        OS: FeatureSpace<Tensor>,
        AS: ParameterizedDistributionSpace<Tensor>,
        PB: ModuleBuilder<P>,
        VB: StepValueBuilder<V>,
        V: StepValue,
    {
        PolicyValueNetActor::new(env, self, policy_vs, value_vs)
    }
}

/// Policy and Value Network Actor
///
/// Actor that maintains a policy and a value network.
/// Takes actions and records history.
///
/// Accepts a callback function to update the model parameters.
pub struct PolicyValueNetActor<OS, AS, P, V>
where
    OS: Space,
    AS: Space,
{
    /// Environment observation space
    pub observation_space: OS,

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

    /// The policy module.
    pub policy: P,

    /// The value estimator module.
    pub value: V,

    /// The recorded step history.
    history: HistoryBuffer<OS::Element, AS::Element>,
}

impl<OS, AS, P, V> PolicyValueNetActor<OS, AS, P, V>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedDistributionSpace<Tensor>,
    V: StepValue,
{
    /// Create a new `PolicyValueNetActor`
    ///
    /// # Args
    /// * `env` - Environment structure.
    /// * `config` - `PolicyValueNetActor` configuration parameters.
    /// * `policy_vs` - Path in which the policy network variables are stored
    /// * `value_vs` - Path in which the value network variables are stored.
    pub fn new<PB, VB>(
        env: EnvStructure<OS, AS>,
        config: &PolicyValueNetActorConfig<PB, VB>,
        policy_vs: &Path,
        value_vs: &Path,
    ) -> Self
    where
        PB: ModuleBuilder<P>,
        VB: StepValueBuilder<V>,
    {
        let observation_space = env.observation_space;
        let action_space = env.action_space;
        let max_steps_per_epoch = config.steps_per_epoch + config.steps_per_epoch / 10;

        let policy = config.policy_config.build_module(
            policy_vs,
            observation_space.num_features(),
            action_space.num_distribution_params(),
        );

        let value = config
            .value_config
            .build_step_value(value_vs, observation_space.num_features());
        let discount_factor = value.discount_factor(env.discount_factor);

        Self {
            observation_space,
            action_space,
            discount_factor,
            steps_per_epoch: config.steps_per_epoch,
            max_steps_per_epoch,
            value_train_iters: config.value_train_iters,
            policy,
            value,
            history: HistoryBuffer::new(Some(max_steps_per_epoch)),
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
        let observation_features = self.observation_space.features(observation);

        if new_episode {
            self.policy.reset();
        }
        tch::no_grad(|| {
            let output = self.policy.step(&observation_features);
            self.action_space.sample_element(&output)
        })
    }
}

impl<OS, AS, P, V> PolicyValueNetActor<OS, AS, P, V>
where
    OS: Space,
    AS: Space,
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
        logger: &mut dyn Logger,
    ) where
        F: FnOnce(&Self, &LazyPackedHistoryFeatures<OS, AS>, &mut dyn Logger) -> Tensor,
        G: FnMut(&Self, &LazyPackedHistoryFeatures<OS, AS>, &mut dyn Logger) -> Tensor,
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
    ///     and returns an estimate of the policy entropy.
    ///
    /// * `update_value` - Callback function that performs a value net update
    ///     and returns the value function loss (pre- or post-update).
    ///     Will be called `value_train_iters` times if `value.trainable()` is true.
    ///
    /// * `logger` - Logger to which epoch statistics are logged. Forwarded to the callbacks.

    fn epoch_update<F, G>(&mut self, update_policy: F, mut update_value: G, logger: &mut dyn Logger)
    where
        F: FnOnce(&Self, &LazyPackedHistoryFeatures<OS, AS>, &mut dyn Logger) -> Tensor,
        G: FnMut(&Self, &LazyPackedHistoryFeatures<OS, AS>, &mut dyn Logger) -> Tensor,
    {
        let features = LazyPackedHistoryFeatures::new(
            self.history.steps(),
            self.history.episode_ranges(),
            &self.observation_space,
            &self.action_space,
            self.discount_factor,
        );

        let entropy = update_policy(self, &features, logger);

        if self.value.trainable() {
            for i in 0..self.value_train_iters {
                let value_loss = update_value(self, &features, logger);

                if i == 0 {
                    epoch_log_scalar(logger, "value_loss_initial", &value_loss);
                } else if i == self.value_train_iters - 1 {
                    epoch_log_scalar(logger, "value_loss_final", &value_loss);
                }
            }
        }

        epoch_log_scalar(logger, "num_steps", self.history.len() as f64);
        epoch_log_scalar(logger, "num_episodes", self.history.num_episodes() as f64);
        epoch_log_scalar(logger, "policy_entropy", &entropy);
        logger.done(Event::Epoch);

        self.history.clear();
    }
}

/// Log a value with the epoch event.
fn epoch_log_scalar<L, V>(logger: &mut L, name: &str, value: V)
where
    L: Logger + ?Sized,
    V: Into<f64>,
{
    logger.log(Event::Epoch, name, value.into().into()).unwrap();
}
