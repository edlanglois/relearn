//! Vanilla SequenceModule + StatefulIterativeModule Gradient with Generalized Advantage Estimation
use super::super::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use super::history::{
    features, HistoryBuffer, LazyPackedHistoryFeatures, PackedHistoryFeaturesView,
};
use super::step_value::{StepValue, StepValueBuilder};
use crate::logging::{Event, Logger};
use crate::spaces::{FeatureSpace, ParameterizedSampleSpace, Space};
use crate::torch::seq_modules::{SequenceModule, StatefulIterativeModule};
use crate::torch::{ModuleBuilder, Optimizer, OptimizerBuilder};
use crate::EnvStructure;
use tch::{kind::Kind, nn, Device, Tensor};

/// Configuration for [PolicyGradientAgent]
#[derive(Debug)]
pub struct PolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub steps_per_epoch: usize,
    pub value_train_iters: u64,
    pub policy_config: PB,
    pub policy_optimizer_config: POB,
    pub value_config: VB,
    pub value_optimizer_config: VOB,
}

impl<PB, POB, VB, VOB> PolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub fn new(
        steps_per_epoch: usize,
        value_train_iters: u64,
        policy_config: PB,
        policy_optimizer_config: POB,
        value_config: VB,
        value_optimizer_config: VOB,
    ) -> Self {
        Self {
            steps_per_epoch,
            value_train_iters,
            policy_config,
            policy_optimizer_config,
            value_config,
            value_optimizer_config,
        }
    }
}

impl<PB, POB, VB, VOB> Default for PolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    PB: Default,
    POB: Default,
    VB: Default,
    VOB: Default,
{
    fn default() -> Self {
        Self {
            steps_per_epoch: 1000,
            value_train_iters: 80,
            policy_config: Default::default(),
            policy_optimizer_config: Default::default(),
            value_config: Default::default(),
            value_optimizer_config: Default::default(),
        }
    }
}

impl<OS, AS, PB, P, POB, PO, VB, V, VOB, VO>
    AgentBuilder<PolicyGradientAgent<OS, AS, P, PO, V, VO>, OS, AS>
    for PolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    PB: ModuleBuilder<P>,
    P: SequenceModule + StatefulIterativeModule,
    POB: OptimizerBuilder<PO>,
    VB: StepValueBuilder<V>,
    V: StepValue,
    VOB: OptimizerBuilder<VO>,
{
    fn build_agent(
        &self,
        env: EnvStructure<OS, AS>,
        _seed: u64,
    ) -> Result<PolicyGradientAgent<OS, AS, P, PO, V, VO>, BuildAgentError> {
        Ok(PolicyGradientAgent::new(
            env.observation_space,
            env.action_space,
            env.discount_factor,
            self.steps_per_epoch,
            self.value_train_iters,
            &self.policy_config,
            &self.policy_optimizer_config,
            &self.value_config,
            &self.value_optimizer_config,
        ))
    }
}

/// SequenceModule + StatefulIterativeModule Gradient with Generalized Advantage Estimation
///
/// Supports both recurrent and non-recurrent policies.
///
/// Reference:
/// High-Dimensional Continuous Control Using Generalized Advantage Estimation
/// by Schulman et al. (2016)
pub struct PolicyGradientAgent<OS, AS, P, PO, V, VO>
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

    /// Number of step value update iterations per epoch.
    pub value_train_iters: u64,

    /// Maximum number of steps per epoch.
    ///
    /// The actual number of steps is the first episode end
    /// between `steps_per_epoch` and `max_steps_per_epoch`,
    /// or `max_steps_per_epoch` if no episode end is found.
    pub max_steps_per_epoch: usize,

    /// When training, only include steps where the discount factor on unknown return is <= this.
    ///
    /// If the discount factor is `d` and `n` steps are observed until the episode ends at
    /// a non-terminal state then the unknown return discount is d^n.
    pub max_unknown_return_discount: f64,

    /// The policy module.
    pub policy: P,

    /// SequenceModule + StatefulIterativeModule optimizer
    policy_optimizer: PO,

    /// The value estimator module.
    pub value: V,

    /// Step value module optimizer.
    value_optimizer: VO,

    /// The recorded step history.
    history: HistoryBuffer<OS::Element, AS::Element>,
}

impl<OS, AS, P, PO, V, VO> PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    V: StepValue,
{
    pub fn new<PB, VB, POB, VOB>(
        observation_space: OS,
        action_space: AS,
        env_discount_factor: f64,
        steps_per_epoch: usize,
        value_train_iters: u64,
        policy_config: &PB,
        policy_optimizer_config: &POB,
        value_config: &VB,
        value_optimizer_config: &VOB,
    ) -> Self
    where
        PB: ModuleBuilder<P>,
        VB: StepValueBuilder<V>,
        POB: OptimizerBuilder<PO>,
        VOB: OptimizerBuilder<VO>,
    {
        let max_steps_per_epoch = (steps_per_epoch as f64 * 1.1) as usize;

        let policy_vs = nn::VarStore::new(Device::Cpu);
        let policy = policy_config.build_module(
            &policy_vs.root(),
            observation_space.num_features(),
            action_space.num_sample_params(),
        );
        let policy_optimizer = policy_optimizer_config.build_optimizer(&policy_vs).unwrap();

        let value_vs = nn::VarStore::new(Device::Cpu);
        let value =
            value_config.build_step_value(&value_vs.root(), observation_space.num_features());
        let value_optimizer = value_optimizer_config.build_optimizer(&value_vs).unwrap();

        let discount_factor = value.discount_factor(env_discount_factor);

        Self {
            observation_space,
            action_space,
            discount_factor,
            steps_per_epoch,
            value_train_iters,
            max_steps_per_epoch: (steps_per_epoch as f64 * 1.1) as usize,
            max_unknown_return_discount: 0.1,
            policy,
            policy_optimizer,
            value,
            value_optimizer,
            history: HistoryBuffer::new(Some(max_steps_per_epoch)),
        }
    }
}

impl<OS, AS, P, PO, V, VO> Actor<OS::Element, AS::Element>
    for PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: StatefulIterativeModule,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        let observation_features = self.observation_space.features(observation);

        if new_episode {
            self.policy.reset();
        }
        tch::no_grad(|| {
            let output = self.policy.step(&observation_features);
            ParameterizedSampleSpace::sample(&self.action_space, &output)
        })
    }
}

impl<OS, AS, P, PO, V, VO> Agent<OS::Element, AS::Element>
    for PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: SequenceModule + StatefulIterativeModule,
    PO: Optimizer,
    V: StepValue,
    VO: Optimizer,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        Actor::act(self, observation, new_episode)
    }

    fn update(&mut self, step: Step<OS::Element, AS::Element>, logger: &mut dyn Logger) {
        let episode_done = step.episode_done;
        self.history.push(step);

        let history_len = self.history.len();
        if history_len < self.steps_per_epoch
            || (history_len < self.max_steps_per_epoch && !episode_done)
        {
            return;
        }

        self.epoch_update(logger);
    }
}

impl<OS, AS, P, PO, V, VO> PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: SequenceModule,
    PO: Optimizer,
    V: StepValue,
    VO: Optimizer,
{
    /// Perform an epoch update: update the policy and step value module and clear history.
    fn epoch_update(&mut self, logger: &mut dyn Logger) {
        let num_steps = self.history.len();
        let num_episodes = self.history.num_episodes();

        let features = LazyPackedHistoryFeatures::new(
            self.history.steps(),
            self.history.episode_ranges(),
            &self.observation_space,
            self.discount_factor,
        );
        let step_values = tch::no_grad(|| self.value.seq_packed(&features));

        let policy_output = self
            .policy
            .seq_packed(features.observations(), features.batch_sizes_tensor());

        for i in 0..self.value_train_iters {
            let value_loss = match self.value.loss(&features) {
                None => break, // no trainable variables
                Some(value_loss) => value_loss,
            };

            if i == 0 {
                logger
                    .log(
                        Event::Epoch,
                        "value_loss_initial",
                        f64::from(&value_loss).into(),
                    )
                    .unwrap();
            } else if i == self.value_train_iters - 1 {
                logger
                    .log(
                        Event::Epoch,
                        "value_loss_final",
                        f64::from(&value_loss).into(),
                    )
                    .unwrap();
            }

            self.value_optimizer.backward_step(&value_loss);
        }

        let features = features.finalize();

        // Consume steps into a vector of actions.
        let (drain_steps, _) = self.history.drain();
        let actions = features::into_packed_actions(drain_steps, &features.episode_ranges);

        let (log_probs, entropies) = self.action_space.batch_statistics(&policy_output, &actions);

        let policy_loss = -(log_probs * &step_values).mean(Kind::Float);
        self.policy_optimizer.backward_step(&policy_loss);

        logger
            .log(Event::Epoch, "batch_num_steps", (num_steps as f64).into())
            .unwrap();
        logger
            .log(
                Event::Epoch,
                "batch_num_episodes",
                (num_episodes as f64).into(),
            )
            .unwrap();
        logger
            .log(
                Event::Epoch,
                "policy_entropy",
                f64::from(entropies.mean(Kind::Float)).into(),
            )
            .unwrap();
        logger
            .log(
                Event::Epoch,
                "step_values",
                f64::from(step_values.mean(Kind::Float)).into(),
            )
            .unwrap();
        logger.done(Event::Epoch);
    }
}

#[cfg(test)]
mod gae_policy_gradient {
    use super::super::super::testing;
    use super::super::step_value::{Gae, GaeConfig, Return};
    use super::*;
    use crate::torch::modules::MlpConfig;
    use crate::torch::optimizers::AdamConfig;
    use crate::torch::seq_modules::{GruMlp, RnnMlpConfig, WithState};
    use tch::nn::Sequential;

    fn test_train_default_policy_gradient<P, PB, V, VB>()
    where
        P: SequenceModule + StatefulIterativeModule,
        PB: ModuleBuilder<P> + Default,
        V: StepValue,
        VB: StepValueBuilder<V> + Default,
    {
        let mut config = PolicyGradientAgentConfig::<PB, AdamConfig, VB, AdamConfig>::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.policy_optimizer_config.learning_rate = 0.1;
        config.value_optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| -> PolicyGradientAgent<_, _, P, _, V, _> {
                config.build_agent(env_structure, 0).unwrap()
            },
            1_000,
            0.9,
        );
    }

    #[test]
    fn default_mlp_return_learns_derministic_bandit() {
        test_train_default_policy_gradient::<Sequential, MlpConfig, Return, Return>()
    }

    #[test]
    fn default_mlp_gae_mlp_learns_derministic_bandit() {
        test_train_default_policy_gradient::<
            Sequential,
            MlpConfig,
            Gae<Sequential>,
            GaeConfig<MlpConfig>,
        >()
    }

    #[test]
    fn default_gru_mlp_return_learns_derministic_bandit() {
        test_train_default_policy_gradient::<WithState<GruMlp>, RnnMlpConfig, Return, Return>()
    }

    #[test]
    fn default_gru_mlp_gae_mlp_derministic_bandit() {
        test_train_default_policy_gradient::<
            WithState<GruMlp>,
            RnnMlpConfig,
            Gae<Sequential>,
            GaeConfig<MlpConfig>,
        >()
    }

    #[test]
    fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
        test_train_default_policy_gradient::<
            WithState<GruMlp>,
            RnnMlpConfig,
            Gae<WithState<GruMlp>>,
            GaeConfig<RnnMlpConfig>,
        >()
    }
}
