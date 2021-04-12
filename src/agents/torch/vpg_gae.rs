//! Vanilla SequenceModule + StatefulIterativeModule Gradient with Generalized Advantage Estimation
use super::super::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use crate::logging::{Event, Logger};
use crate::spaces::{FeatureSpace, ParameterizedSampleSpace, Space};
use crate::torch::seq_modules::{SequenceModule, StatefulIterativeModule};
use crate::torch::{ModuleBuilder, Optimizer, OptimizerBuilder};
use crate::EnvStructure;
use std::iter;
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
    history: Vec<Step<OS::Element, AS::Element>>,
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
            // Slight excess in case of off-by-one errors.
            history: Vec::with_capacity(max_steps_per_epoch + 1),
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
        let mut episode_length = 0;
        let episode_lengths: Vec<_> = self
            .history
            .iter()
            .filter_map(|step| {
                episode_length += 1;
                if step.episode_done {
                    let len = episode_length;
                    episode_length = 0;
                    Some(len)
                } else {
                    None
                }
            })
            .collect();

        let observation_features = self
            .observation_space
            .batch_features(self.history.iter().map(|step| &step.observation));
        let batched_observation_features = observation_features.unsqueeze(0);

        let estimated_values = self
            .value_fn
            .seq_serial(&batched_observation_features, &episode_lengths)
            .squeeze1(0);

        // Tensor::iter looks slow, insert into a vector first
        let estimated_values_vec: Vec<_> = estimated_values.into();
        let residuals_with_next_value: Vec<_> = self
            .history
            .iter()
            .zip(estimated_values_vec.iter())
            .zip(
                estimated_values_vec
                    .iter()
                    .skip(1)
                    .map(Some)
                    .chain(iter::once(None)),
            )
            .map(|((step, &value), next_value)| {
                let next_value = match (&step.next_observation, next_value) {
                    (None, _) => 0.0,                         // terminal
                    (_, Some(&v)) if !step.episode_done => v, // episode continues
                    (Some(_next_obs), _) => {
                        // episode done with a non-terminal step
                        // should use the estimated value of next_obs
                        // which may require iterating over the whole episode
                        panic!("non-terminal end-of-episode not yet supported");
                    }
                };
                let residual = step.reward + self.discount_factor * next_value - value;
                (residual, next_value)
            })
            .collect();

        let advantage_discount = self.lambda * self.discount_factor;
        let mut next_advantage = 0.0;
        let mut next_return = 0.0;
        let (advantages, returns): (Vec<_>, Vec<_>) = self
            .history
            .iter()
            .rev()
            .zip(residuals_with_next_value.into_iter().rev())
            .map(|(step, (residual, estimated_next_value))| {
                if step.episode_done {
                    next_advantage = 0.0;
                    next_return = estimated_next_value;
                } else {
                    next_advantage *= advantage_discount;
                }
                next_return *= self.discount_factor;

                next_advantage += residual;
                next_return += step.reward;
                (next_advantage as f32, next_return as f32)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .unzip();
        let advantages = Tensor::of_slice(&advantages);
        let returns = Tensor::of_slice(&returns);

        let actions: Vec<_> = self.history.drain(..).map(|step| step.action).collect();

        let policy_output = self
            .policy
            .seq_serial(&batched_observation_features, &episode_lengths)
            .squeeze1(0);
        let (log_probs, entropies) = self.action_space.batch_statistics(&policy_output, &actions);

        let policy_loss = -(log_probs * advantages).mean(Kind::Float);
        self.policy_optimizer.backward_step(&policy_loss);

        for i in 0..self.value_fn_train_iters {
            let value_fn_loss = self
                .value_fn
                .seq_serial(&batched_observation_features, &episode_lengths)
                .squeeze1(-1) // feature dim
                .squeeze1(0) // batch dim
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
            .log(
                Event::Epoch,
                "batch_num_steps",
                (actions.len() as f64).into(),
            )
            .unwrap();
        logger
            .log(
                Event::Epoch,
                "batch_num_episodes",
                (episode_lengths.len() as f64).into(),
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
