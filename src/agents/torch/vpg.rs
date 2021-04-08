//! Vanilla Policy Gradient
use super::super::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use super::Policy;
use crate::logging::{Event, Logger};
use crate::spaces::{FeatureSpace, ParameterizedSampleSpace, Space};
use crate::torch::seq_modules::IterativeModule;
use crate::torch::{ModuleBuilder, Optimizer, OptimizerBuilder};
use crate::EnvStructure;
use tch::{kind::Kind, nn, Device, Tensor};

/// Configuration for PolicyGradientAgent
#[derive(Debug)]
pub struct PolicyGradientAgentConfig<PB, OB>
where
    PB: ModuleBuilder,
    <PB as ModuleBuilder>::Module: Policy,
    OB: OptimizerBuilder,
{
    pub steps_per_epoch: usize,
    pub policy_config: PB,
    pub optimizer_config: OB,
}

impl<PB, OB> PolicyGradientAgentConfig<PB, OB>
where
    PB: ModuleBuilder,
    <PB as ModuleBuilder>::Module: Policy,
    OB: OptimizerBuilder,
{
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
    PB: ModuleBuilder + Default,
    <PB as ModuleBuilder>::Module: Policy,
    OB: OptimizerBuilder + Default,
{
    fn default() -> Self {
        Self {
            steps_per_epoch: 4000,
            policy_config: Default::default(),
            optimizer_config: Default::default(),
        }
    }
}

impl<OS, AS, PB, OB> AgentBuilder<OS, AS> for PolicyGradientAgentConfig<PB, OB>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    PB: ModuleBuilder,
    <PB as ModuleBuilder>::Module: Policy,
    OB: OptimizerBuilder,
{
    type Agent = PolicyGradientAgent<OS, AS, <PB as ModuleBuilder>::Module, OB::Optimizer>;

    fn build(&self, es: EnvStructure<OS, AS>, _seed: u64) -> Result<Self::Agent, BuildAgentError> {
        Ok(Self::Agent::new(
            es.observation_space,
            es.action_space,
            es.discount_factor,
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
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: IterativeModule,
    O: Optimizer,
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

    /// The policy recurrent neural network.
    pub policy: P,

    /// The RNN policy hidden state.
    state: P::State,

    /// The recorded step history.
    history: Vec<Step<OS::Element, AS::Element>>,

    /// Optimizer
    optimizer: O,
}

impl<OS, AS, P, O> PolicyGradientAgent<OS, AS, P, O>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: IterativeModule,
    O: Optimizer,
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
        PB: ModuleBuilder<Module = P>,
        OB: OptimizerBuilder<Optimizer = O>,
    {
        let max_steps_per_epoch = (steps_per_epoch as f64 * 1.1) as usize;
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = policy_config.build(
            &vs.root(),
            observation_space.num_features(),
            action_space.num_sample_params(),
        );
        let state = policy.initial_state(1);
        let optimizer = optimizer_config.build(&vs).unwrap();
        Self {
            observation_space,
            action_space,
            discount_factor,
            steps_per_epoch,
            max_steps_per_epoch: (steps_per_epoch as f64 * 1.1) as usize,
            max_unknown_return_discount: 0.1,
            policy,
            state,
            // Slight excess in case of off-by-one errors.
            history: Vec::with_capacity(max_steps_per_epoch + 1),
            optimizer,
        }
    }
}

impl<OS, AS, P, O> Actor<OS::Element, AS::Element> for PolicyGradientAgent<OS, AS, P, O>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: IterativeModule,
    O: Optimizer,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        let observation_features = self.observation_space.features(observation).unsqueeze(0);

        if new_episode {
            self.state = self.policy.initial_state(1);
        }
        let state = &self.state;

        let (action, state) = tch::no_grad(|| {
            let (output, state) = self.policy.step(&observation_features, &state);
            let output = output.squeeze1(0); // Squeeze the batch dimension
            let action = ParameterizedSampleSpace::sample(&self.action_space, &output);
            (action, state)
        });
        self.state = state;
        action
    }
}

impl<OS, AS, P, O> Agent<OS::Element, AS::Element> for PolicyGradientAgent<OS, AS, P, O>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedSampleSpace<Tensor>,
    P: Policy,
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
        let history_data: HistoryData<AS> = tch::no_grad(|| {
            history_data(
                &mut self.history,
                &self.observation_space,
                self.discount_factor,
                self.max_unknown_return_discount,
            )
        });

        let output = self
            .policy
            .seq_serial(
                &history_data.observation_features.unsqueeze(0), // Batch dimension
                &history_data.episode_lengths,
            )
            .squeeze1(0);
        let (log_probs, entropies) = self
            .action_space
            .batch_statistics(&output, &history_data.actions);

        let loss = -(log_probs * history_data.returns).mean(Kind::Float);
        self.optimizer.backward_step(&loss);

        logger
            .log(
                Event::Epoch,
                "batch_num_steps",
                (history_data.actions.len() as f64).into(),
            )
            .unwrap();
        logger
            .log(
                Event::Epoch,
                "batch_num_episodes",
                (history_data.episode_lengths.len() as f64).into(),
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

// TODO: Do something like packed sequence so that batch processing is possible.
struct HistoryData<AS: Space> {
    /// Observation features. A f32 tensor of shape [NUM_STEPS, NUM_OBS_FEATURES]
    observation_features: Tensor,
    /// Actions.
    actions: Vec<AS::Element>,
    // /// Log probabilities of the selected actions. A f32 tensor of shape [NUM_STEPS]
    // action_log_probs: Tensor,
    /// Cumulative discounted observed episode returns. A f32 tensor of shape [NUM_STEPS]
    returns: Tensor,
    /// The number of steps in each episode as ordered along the NUM_STEPS dimension.
    episode_lengths: Vec<usize>,
}

/// Convert a step history vector into data tensors / vectors.
///
/// Empties history in the process.
///
/// An episode can end with a terminal or a non terminal state.
/// If an episode ends with a terminal state then all steps are included in the result.
///
/// If an episode ends with a non-terminal state then there may be future unobserved rewards in the
/// episode so an estimate of the return using just the observed steps will be unreliable.
/// Therefore, when an episode ends in a non-terminal state, the steps close to the end of the
/// episode are dropped. Specifically a step is dropped if the discounting applied to the unknown
/// part of the return is > max_unknown_return_discount.
fn history_data<OS: FeatureSpace<Tensor>, AS: ParameterizedSampleSpace<Tensor>>(
    history: &mut Vec<Step<OS::Element, AS::Element>>,
    observation_space: &OS,
    discount_factor: f64,
    max_unknown_return_discount: f64,
) -> HistoryData<AS> {
    // For each step, calculate state = (known_return, unknown_return_discount)
    // * known_return is the discounted sum of rewards in all future observed steps of the episode.
    // * unknown_return_scale is the discounting applied to any unknown rewards that might appear
    //      in the episode after the observed steps.
    //      This is 0 if all steps of the spisode are observed.
    //
    // Then include steps that have unknown_return_discount < max_unknown_return_discount
    let (known_returns, include_steps): (Vec<f64>, Vec<bool>) = history
        .iter()
        .rev()
        .scan((0.0, 1.0), |state, step| {
            if step.next_observation.is_none() {
                // Terminal state
                *state = (0.0, 0.0)
            } else if step.episode_done {
                // Non-terminal end of episode
                *state = (0.0, 1.0)
            }
            state.0 *= discount_factor;
            state.1 *= discount_factor;
            state.0 += step.reward;
            Some((state.0, state.1 <= max_unknown_return_discount))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .unzip();

    let mut episode_lengths: Vec<usize> = Vec::new();
    let mut episode_length = 0;
    for (step, &include) in history.iter().zip(include_steps.iter()) {
        if include {
            episode_length += 1
        }
        if step.episode_done && episode_length > 0 {
            episode_lengths.push(episode_length);
            episode_length = 0;
        }
    }

    assert!(
        episode_lengths.len() > 0,
        "No steps were included in the training batch. \
        This can happen if the discount factor (={}) is close to 1 \
        and episodes never end or are cut off before reaching a terminal state.",
        discount_factor
    );

    // A f32 tensor of shape [NUM_STEPS] containing cumulative known episode returns for each step.
    let returns = Tensor::of_slice(
        &known_returns
            .into_iter()
            .zip(include_steps.iter())
            .filter_map(|(return_, &include)| if include { Some(return_ as f32) } else { None })
            .collect::<Vec<_>>(),
    );

    let observations = history
        .iter()
        .zip(include_steps.iter())
        .filter_map(|(step, &include)| {
            if include {
                Some(&step.observation)
            } else {
                None
            }
        });
    // A tensor of shape [NUM_STEPS, NUM_OBSERVATION_FEATURES] containing one-hot feature vectors.
    let observation_features = observation_space.batch_features(observations);

    let actions = history
        .drain(..)
        .zip(include_steps)
        .filter_map(|(step, include)| if include { Some(step.action) } else { None })
        .collect();

    HistoryData {
        observation_features,
        actions,
        returns,
        episode_lengths,
    }
}

#[cfg(test)]
mod policy_gradient {
    use super::super::super::testing;
    use super::*;
    use crate::torch::configs::{GruMlpConfig, MlpConfig};
    use crate::torch::optimizers::AdamConfig;

    #[test]
    fn default_mlp_learns_derministic_bandit() {
        let mut config = PolicyGradientAgentConfig::<MlpConfig, AdamConfig>::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| config.build(env_structure, 0).unwrap(),
            1_000,
            0.9,
        );
    }

    #[test]
    fn default_gru_mlp_learns_derministic_bandit() {
        let mut config = PolicyGradientAgentConfig::<GruMlpConfig, AdamConfig>::default();
        // Speed up learning for this simple environment
        config.steps_per_epoch = 1000;
        config.optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| config.build(env_structure, 0).unwrap(),
            1_000,
            0.9,
        );
    }
}
