//! Vanilla SequenceModule + StatefulIterativeModule Gradient
use super::super::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use super::HistoryBuffer;
use crate::logging::{Event, Logger};
use crate::spaces::{FeatureSpace, ParameterizedSampleSpace, Space};
use crate::torch::seq_modules::{SequenceModule, StatefulIterativeModule};
use crate::torch::{ModuleBuilder, Optimizer, OptimizerBuilder};
use crate::utils::packed::{PackedBatchSizes, PackingIndices};
use crate::EnvStructure;
use tch::{kind::Kind, nn, Device, Tensor};

/// Configuration for PolicyGradientAgent
#[derive(Debug)]
pub struct PolicyGradientAgentConfig<PB, OB> {
    pub steps_per_epoch: usize,
    pub policy_config: PB,
    pub optimizer_config: OB,
}

impl<PB, OB> PolicyGradientAgentConfig<PB, OB> {
    pub fn new(steps_per_epoch: usize, policy_config: PB, optimizer_config: OB) -> Self {
        Self {
            steps_per_epoch,
            policy_config,
            optimizer_config,
        }
    }
}

impl<PB, OB> Default for PolicyGradientAgentConfig<PB, OB>
where
    PB: Default,
    OB: Default,
{
    fn default() -> Self {
        Self {
            steps_per_epoch: 1000,
            policy_config: Default::default(),
            optimizer_config: Default::default(),
        }
    }
}

impl<OS, AS, PB, P, OB, O> AgentBuilder<PolicyGradientAgent<OS, AS, P, O>, OS, AS>
    for PolicyGradientAgentConfig<PB, OB>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    PB: ModuleBuilder<P>,
    OB: OptimizerBuilder<O>,
{
    fn build_agent(
        &self,
        env: EnvStructure<OS, AS>,
        _seed: u64,
    ) -> Result<PolicyGradientAgent<OS, AS, P, O>, BuildAgentError> {
        Ok(PolicyGradientAgent::new(
            env.observation_space,
            env.action_space,
            env.discount_factor,
            self.steps_per_epoch,
            &self.policy_config,
            &self.optimizer_config,
        ))
    }
}

/// A vanilla policy-gradient agent.
///
/// Supports both recurrent and non-recurrent policies.
pub struct PolicyGradientAgent<OS, AS, P, O>
where
    OS: Space,
    AS: Space,
{
    /// Environment observation space
    pub observation_space: OS,

    /// Environment action space
    pub action_space: AS,

    /// Amount by which future rewards are discounted.
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

    /// When training, only include steps where the discount factor on unknown return is <= this.
    ///
    /// If the discount factor is `d` and `n` steps are observed until the episode ends at
    /// a non-terminal state then the unknown return discount is d^n.
    pub max_unknown_return_discount: f64,

    /// The policy module.
    pub policy: P,

    /// The recorded step history.
    history: HistoryBuffer<OS::Element, AS::Element>,

    /// Optimizer
    optimizer: O,
}

impl<OS, AS, P, O> PolicyGradientAgent<OS, AS, P, O>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
{
    pub fn new<PB, OB>(
        observation_space: OS,
        action_space: AS,
        discount_factor: f64,
        steps_per_epoch: usize,
        policy_config: &PB,
        optimizer_config: &OB,
    ) -> Self
    where
        PB: ModuleBuilder<P>,
        OB: OptimizerBuilder<O>,
    {
        let max_steps_per_epoch = (steps_per_epoch as f64 * 1.1) as usize;
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = policy_config.build_module(
            &vs.root(),
            observation_space.num_features(),
            action_space.num_sample_params(),
        );
        let optimizer = optimizer_config.build_optimizer(&vs).unwrap();
        Self {
            observation_space,
            action_space,
            discount_factor,
            steps_per_epoch,
            max_steps_per_epoch: (steps_per_epoch as f64 * 1.1) as usize,
            max_unknown_return_discount: 0.1,
            policy,
            history: HistoryBuffer::new(Some(max_steps_per_epoch)),
            optimizer,
        }
    }
}

impl<OS, AS, P, O> Actor<OS::Element, AS::Element> for PolicyGradientAgent<OS, AS, P, O>
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
            self.action_space.sample(&output)
        })
    }
}

impl<OS, AS, P, O> Agent<OS::Element, AS::Element> for PolicyGradientAgent<OS, AS, P, O>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: SequenceModule + StatefulIterativeModule,
    O: Optimizer,
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

        // Epoch
        // Update the policy

        // TODO: extract feature calculation to function and set no_grad
        // let _no_grad = tch::no_grad_guard();

        let num_steps = self.history.len();
        let num_episodes = self.history.num_episodes();

        let steps = self.history.steps();
        let mut episode_ranges: Vec<_> = self.history.episode_ranges().collect();
        // Sort in decreasing order of length; required for packing
        episode_ranges.sort_by(|a, b| a.len().cmp(&b.len()).reverse());

        // Packed observation features
        let observation_features = self.observation_space.batch_features(
            PackingIndices::from_sorted(&episode_ranges).map(|i| &steps[i].observation),
        );

        // Step returns in the reverse order as steps
        let step_returns_rev: Vec<_> = steps
            .iter()
            .rev()
            .scan(0.0, |next_return, step| {
                if step.next_observation.is_none() {
                    // Terminal state
                    *next_return = 0.0
                } else if step.episode_done {
                    // Non-terminal end-of-episode
                    panic!("Non-terminal end-of-episode not currently supported");
                }
                *next_return *= self.discount_factor;
                *next_return += step.reward;
                Some(*next_return as f32)
            })
            .collect();
        // Packed returns
        let returns: Vec<_> = PackingIndices::from_sorted(&episode_ranges)
            .map(|i| step_returns_rev[num_steps - 1 - i])
            .collect();
        let returns = Tensor::of_slice(&returns);

        // Actions are not necessarily copyable to drain steps and take the actions.
        // Put into Option so that we can take the action to when packing.
        let (drain_steps, _) = self.history.drain();
        let mut seq_actions: Vec<_> = drain_steps.map(|step| Some(step.action)).collect();
        // Packed actions
        let actions: Vec<_> = PackingIndices::from_sorted(&episode_ranges)
            .map(|i| seq_actions[i].take().unwrap())
            .collect();

        let batch_sizes: Vec<_> = PackedBatchSizes::from_sorted_ranges(&episode_ranges)
            .map(|s| s as i64)
            .collect();
        let batch_sizes = Tensor::of_slice(&batch_sizes);

        let output = self.policy.seq_packed(&observation_features, &batch_sizes);
        let (log_probs, entropies) = self.action_space.batch_statistics(&output, &actions);

        let loss = -(log_probs * returns).mean(Kind::Float);
        self.optimizer.backward_step(&loss);

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
}

#[cfg(test)]
mod policy_gradient {
    use super::super::super::{testing, AgentBuilder};
    use super::*;
    use crate::torch::configs::{MlpConfig, RnnMlpConfig};
    use crate::torch::optimizers::AdamConfig;
    use crate::torch::seq_modules::{GruMlp, WithState};
    use tch::nn::Sequential;

    #[test]
    fn default_mlp_learns_derministic_bandit() {
        let mut config = PolicyGradientAgentConfig::<MlpConfig, AdamConfig>::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| -> PolicyGradientAgent<_, _, Sequential, _> {
                config.build_agent(env_structure, 0).unwrap()
            },
            1_000,
            0.9,
        );
    }

    #[test]
    fn default_gru_mlp_learns_derministic_bandit() {
        let mut config = PolicyGradientAgentConfig::<RnnMlpConfig, AdamConfig>::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| -> PolicyGradientAgent<_, _, WithState<GruMlp>, _> {
                config.build_agent(env_structure, 0).unwrap()
            },
            1_000,
            0.9,
        );
    }
}
