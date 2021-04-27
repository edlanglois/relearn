//! Vanilla SequenceModule + StatefulIterativeModule Gradient with Generalized Advantage Estimation
use super::super::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use super::history::{features, HistoryBuffer};
use crate::logging::{Event, Logger};
use crate::spaces::{FeatureSpace, ParameterizedSampleSpace, Space};
use crate::torch::seq_modules::{SequenceModule, StatefulIterativeModule};
use crate::torch::{ModuleBuilder, Optimizer, OptimizerBuilder};
use crate::utils::packed;
use crate::EnvStructure;
use tch::{kind::Kind, nn, Device, Reduction, Tensor};

/// Configuration for GaePolicyGradientAgent
#[derive(Debug)]
pub struct GaePolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub gamma: f64,
    pub lambda: f64,
    pub steps_per_epoch: usize,
    pub value_fn_train_iters: u64,
    pub policy_config: PB,
    pub policy_optimizer_config: POB,
    pub value_fn_config: VB,
    pub value_fn_optimizer_config: VOB,
}

impl<PB, POB, VB, VOB> GaePolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub fn new(
        gamma: f64,
        lambda: f64,
        steps_per_epoch: usize,
        value_fn_train_iters: u64,
        policy_config: PB,
        policy_optimizer_config: POB,
        value_fn_config: VB,
        value_fn_optimizer_config: VOB,
    ) -> Self {
        Self {
            gamma,
            lambda,
            steps_per_epoch,
            value_fn_train_iters,
            policy_config,
            policy_optimizer_config,
            value_fn_config,
            value_fn_optimizer_config,
        }
    }
}

impl<PB, POB, VB, VOB> Default for GaePolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    PB: Default,
    POB: Default,
    VB: Default,
    VOB: Default,
{
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            steps_per_epoch: 1000,
            value_fn_train_iters: 80,
            policy_config: Default::default(),
            policy_optimizer_config: Default::default(),
            value_fn_config: Default::default(),
            value_fn_optimizer_config: Default::default(),
        }
    }
}

impl<OS, AS, PB, P, POB, PO, VB, V, VOB, VO>
    AgentBuilder<GaePolicyGradientAgent<OS, AS, P, PO, V, VO>, OS, AS>
    for GaePolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    PB: ModuleBuilder<P>,
    P: SequenceModule + StatefulIterativeModule,
    POB: OptimizerBuilder<PO>,
    VB: ModuleBuilder<V>,
    V: SequenceModule + StatefulIterativeModule,
    VOB: OptimizerBuilder<VO>,
{
    fn build_agent(
        &self,
        env: EnvStructure<OS, AS>,
        _seed: u64,
    ) -> Result<GaePolicyGradientAgent<OS, AS, P, PO, V, VO>, BuildAgentError> {
        Ok(GaePolicyGradientAgent::new(
            env.observation_space,
            env.action_space,
            env.discount_factor,
            self.gamma,
            self.lambda,
            self.steps_per_epoch,
            self.value_fn_train_iters,
            &self.policy_config,
            &self.policy_optimizer_config,
            &self.value_fn_config,
            &self.value_fn_optimizer_config,
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
pub struct GaePolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: Space,
    AS: Space,
{
    /// Environment observation space
    pub observation_space: OS,

    /// Environment action space
    pub action_space: AS,

    /// Amount by which future rewards are discounted
    ///
    /// This is the minimum of the environment discount factor and
    /// the GAE-lambda regularization discount factor (gamma).
    pub discount_factor: f64,

    /// Advantage interpolation factor between one-step residuals (=0) and full return (=1).
    pub lambda: f64,

    /// Minimum number of steps to collect per epoch.
    ///
    /// This value is exceeded by search for the next episode boundary,
    /// up to a maximum of `max_steps_per_epoch`.
    pub steps_per_epoch: usize,

    /// Number of value function update iterations per epoch.
    pub value_fn_train_iters: u64,

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
    pub value_fn: V,

    /// Value function optimizer.
    value_fn_optimizer: VO,

    /// The recorded step history.
    history: HistoryBuffer<OS::Element, AS::Element>,
}

impl<OS, AS, P, PO, V, VO> GaePolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
{
    pub fn new<PB, VB, POB, VOB>(
        observation_space: OS,
        action_space: AS,
        env_discount_factor: f64,
        gamma: f64,
        lambda: f64,
        steps_per_epoch: usize,
        value_fn_train_iters: u64,
        policy_config: &PB,
        policy_optimizer_config: &POB,
        value_fn_config: &VB,
        value_fn_optimizer_config: &VOB,
    ) -> Self
    where
        PB: ModuleBuilder<P>,
        VB: ModuleBuilder<V>,
        POB: OptimizerBuilder<PO>,
        VOB: OptimizerBuilder<VO>,
    {
        let max_steps_per_epoch = (steps_per_epoch as f64 * 1.1) as usize;
        let discount_factor = env_discount_factor.min(gamma);

        let policy_vs = nn::VarStore::new(Device::Cpu);
        let policy = policy_config.build_module(
            &policy_vs.root(),
            observation_space.num_features(),
            action_space.num_sample_params(),
        );
        let policy_optimizer = policy_optimizer_config.build_optimizer(&policy_vs).unwrap();

        let value_fn_vs = nn::VarStore::new(Device::Cpu);
        let value_fn =
            value_fn_config.build_module(&value_fn_vs.root(), observation_space.num_features(), 1);
        let value_fn_optimizer = value_fn_optimizer_config
            .build_optimizer(&value_fn_vs)
            .unwrap();

        Self {
            observation_space,
            action_space,
            discount_factor,
            lambda,
            steps_per_epoch,
            value_fn_train_iters,
            max_steps_per_epoch: (steps_per_epoch as f64 * 1.1) as usize,
            max_unknown_return_discount: 0.1,
            policy,
            policy_optimizer,
            value_fn,
            value_fn_optimizer,
            history: HistoryBuffer::new(Some(max_steps_per_epoch)),
        }
    }
}

impl<OS, AS, P, PO, V, VO> Actor<OS::Element, AS::Element>
    for GaePolicyGradientAgent<OS, AS, P, PO, V, VO>
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
    for GaePolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: SequenceModule + StatefulIterativeModule,
    PO: Optimizer,
    V: SequenceModule + StatefulIterativeModule,
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

impl<OS, AS, P, PO, V, VO> GaePolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: SequenceModule,
    PO: Optimizer,
    V: SequenceModule + StatefulIterativeModule,
    VO: Optimizer,
{
    /// Perform an epoch update: update the policy and value function and clear history.
    fn epoch_update(&mut self, logger: &mut dyn Logger) {
        let num_steps = self.history.len();
        let num_episodes = self.history.num_episodes();

        let (observation_features, actions, returns, advantages, batch_sizes) =
            self.drain_history_features();

        let policy_output = self.policy.seq_packed(&observation_features, &batch_sizes);
        let (log_probs, entropies) = self.action_space.batch_statistics(&policy_output, &actions);

        let policy_loss = -(log_probs * advantages).mean(Kind::Float);
        self.policy_optimizer.backward_step(&policy_loss);

        for i in 0..self.value_fn_train_iters {
            let value_fn_loss = self
                .value_fn
                .seq_packed(&observation_features, &batch_sizes)
                .squeeze1(-1) // feature dim
                .mse_loss(&returns, Reduction::Mean);

            if i == 0 {
                logger
                    .log(
                        Event::Epoch,
                        "value_fn_loss_initial",
                        f64::from(&value_fn_loss).into(),
                    )
                    .unwrap();
            } else if i == self.value_fn_train_iters - 1 {
                logger
                    .log(
                        Event::Epoch,
                        "value_fn_loss_final",
                        f64::from(&value_fn_loss).into(),
                    )
                    .unwrap();
            }

            self.value_fn_optimizer.backward_step(&value_fn_loss);
        }

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
        logger.done(Event::Epoch);
    }

    /// Drain the stored history into a set of features. Clears the stored history.
    ///
    /// # Returns
    /// * `observation_features` - Packed observation features.
    ///     An f32 tensor of shape [TOTAL_STEPS, NUM_INPUT_FEATURES].
    ///
    /// * `actions` - Packed step actions. A vector of length TOTAL_STEPS.
    ///
    /// * `returns` - Packed step returns. For each step, discount sum of current and future
    ///     rewards. An f32 tensor of shape [TOTAL_STEPS].
    ///
    /// * `advantages` - Packed step action advantages. An f32 tensor of shape [TOTAL_STEPS].
    ///
    /// * `batch_sizes` - The batch size for each packed time step.
    ///     An i64 tensor of shape [MAX_EPISODE_LENGHT].
    ///
    fn drain_history_features(&mut self) -> (Tensor, Vec<AS::Element>, Tensor, Tensor, Tensor) {
        let _no_grad = tch::no_grad_guard();

        let steps = self.history.steps();
        let episode_ranges = features::sorted_episode_ranges(self.history.episode_ranges());

        // Batch sizes in the packing
        let batch_sizes: Vec<usize> = features::packing_batch_sizes(&episode_ranges).collect();
        let batch_sizes_i64: Vec<_> = batch_sizes.iter().map(|&x| x as i64).collect();
        let batch_sizes_tensor = Tensor::of_slice(&batch_sizes_i64);

        let observation_features =
            features::packed_observation_features(steps, &episode_ranges, &self.observation_space);

        let rewards = features::packed_rewards(steps, &episode_ranges);
        let returns = features::packed_returns(&rewards, &batch_sizes_i64, self.discount_factor);

        // Packed estimated values of the observed states
        let estimated_values = self
            .value_fn
            .seq_packed(&observation_features, &batch_sizes_tensor)
            .squeeze1(-1);

        assert!(
            steps
                .iter()
                .all(|s| !s.episode_done || s.next_observation.is_none()),
            "Non-terminal end-of-episode not supported"
        );

        // Packed estimated value for the observed successor states.
        // Assumes that all end-of-episodes are terminal and have value 0.
        //
        // More generally, we should apply the value function to last_step.next_observation.
        // But this is tricky since the value function can be a sequential module and require the
        // state from the rest of the episode.
        let estimated_next_values =
            packed::packed_tensor_push_shift(&estimated_values, &batch_sizes, 0.0);

        // Packed one-step TD residuals.
        let residuals = &rewards + self.discount_factor * estimated_next_values - estimated_values;

        // Packed step action advantages
        let advantages = packed::packed_tensor_discounted_cumsum_from_end(
            &residuals,
            &batch_sizes_i64,
            self.lambda * self.discount_factor,
        );

        let (drain_steps, _) = self.history.drain();
        let actions = features::into_packed_actions(drain_steps, &episode_ranges);

        (
            observation_features,
            actions,
            returns,
            advantages,
            batch_sizes_tensor,
        )
    }
}

#[cfg(test)]
mod gae_policy_gradient {
    use super::super::super::testing;
    use super::*;
    use crate::torch::configs::{MlpConfig, RnnMlpConfig};
    use crate::torch::optimizers::AdamConfig;
    use crate::torch::seq_modules::{GruMlp, WithState};
    use tch::nn::Sequential;

    #[test]
    fn default_mlp_learns_derministic_bandit() {
        let mut config =
            GaePolicyGradientAgentConfig::<MlpConfig, AdamConfig, MlpConfig, AdamConfig>::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.policy_optimizer_config.learning_rate = 0.1;
        config.value_fn_optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| -> GaePolicyGradientAgent<_, _, Sequential, _, Sequential, _> {
                config.build_agent(env_structure, 0).unwrap()
            },
            1_000,
            0.9,
        );
    }

    #[test]
    fn default_gru_mlp_v_mlp_learns_derministic_bandit() {
        let mut config = GaePolicyGradientAgentConfig::<
            RnnMlpConfig,
            AdamConfig,
            MlpConfig,
            AdamConfig,
        >::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.policy_optimizer_config.learning_rate = 0.1;
        config.value_fn_optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| -> GaePolicyGradientAgent<_, _, WithState<GruMlp>, _, Sequential, _> {
                config.build_agent(env_structure, 0).unwrap()
            },
            1_000,
            0.9,
        );
    }

    #[test]
    #[ignore] // Recurrent training is currently very slow
    fn default_gru_mlp_v_gru_mlp_learns_derministic_bandit() {
        let mut config = GaePolicyGradientAgentConfig::<
            RnnMlpConfig,
            AdamConfig,
            RnnMlpConfig,
            AdamConfig,
        >::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.policy_optimizer_config.learning_rate = 0.1;
        config.value_fn_optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| -> GaePolicyGradientAgent<
                _,
                _,
                WithState<GruMlp>,
                _,
                WithState<GruMlp>,
                _,
            > { config.build_agent(env_structure, 0).unwrap() },
            1_000,
            0.9,
        );
    }
}
