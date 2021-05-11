//! Vanilla Policy Gradient
use super::super::history::{LazyPackedHistoryFeatures, PackedHistoryFeaturesView};
use super::super::seq_modules::{SequenceModule, StatefulIterativeModule};
use super::super::step_value::{StepValue, StepValueBuilder};
use super::super::{ModuleBuilder, Optimizer, OptimizerBuilder};
use super::actor::{PolicyValueNetActor, PolicyValueNetActorConfig};
use crate::agents::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use crate::logging::Logger;
use crate::spaces::{FeatureSpace, ParameterizedSampleSpace, ReprSpace, Space};
use crate::EnvStructure;
use std::cell::Cell;
use tch::{kind::Kind, nn, Device, Tensor};

/// Configuration for [PolicyGradientAgent]
#[derive(Debug, Default)]
pub struct PolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub actor_config: PolicyValueNetActorConfig<PB, VB>,
    pub policy_optimizer_config: POB,
    pub value_optimizer_config: VOB,
}

impl<PB, POB, VB, VOB> PolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub fn new(
        actor_config: PolicyValueNetActorConfig<PB, VB>,
        policy_optimizer_config: POB,
        value_optimizer_config: VOB,
    ) -> Self {
        Self {
            actor_config,
            policy_optimizer_config,
            value_optimizer_config,
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
            &self.actor_config,
            &self.policy_optimizer_config,
            &self.value_optimizer_config,
        ))
    }
}

/// Vanilla Policy Gradient Agent
///
/// Supports both recurrent and non-recurrent policies.
pub struct PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: Space,
    AS: Space,
{
    /// Base actor
    actor: PolicyValueNetActor<OS, AS, P, V>,

    /// SequenceModule + StatefulIterativeModule optimizer
    policy_optimizer: PO,

    /// Step value module optimizer.
    value_optimizer: VO,
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
        actor_config: &PolicyValueNetActorConfig<PB, VB>,
        policy_optimizer_config: &POB,
        value_optimizer_config: &VOB,
    ) -> Self
    where
        PB: ModuleBuilder<P>,
        VB: StepValueBuilder<V>,
        POB: OptimizerBuilder<PO>,
        VOB: OptimizerBuilder<VO>,
    {
        let policy_vs = nn::VarStore::new(Device::Cpu);
        let value_vs = nn::VarStore::new(Device::Cpu);
        let actor = actor_config.build_actor(
            observation_space,
            action_space,
            env_discount_factor,
            &policy_vs.root(),
            &value_vs.root(),
        );

        let policy_optimizer = policy_optimizer_config.build_optimizer(&policy_vs).unwrap();
        let value_optimizer = value_optimizer_config.build_optimizer(&value_vs).unwrap();

        Self {
            actor,
            policy_optimizer,
            value_optimizer,
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
        self.actor.act(observation, new_episode)
    }
}

impl<OS, AS, P, PO, V, VO> Agent<OS::Element, AS::Element>
    for PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ReprSpace<Tensor> + ParameterizedSampleSpace<Tensor>,
    P: SequenceModule + StatefulIterativeModule,
    PO: Optimizer,
    V: StepValue,
    VO: Optimizer,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        Actor::act(self, observation, new_episode)
    }

    fn update(&mut self, step: Step<OS::Element, AS::Element>, logger: &mut dyn Logger) {
        let ref policy_optimizer = self.policy_optimizer;
        let ref value_optimizer = self.value_optimizer;
        self.actor.update(
            step,
            |actor, features, _logger| policy_gradient_update(actor, features, policy_optimizer),
            |actor, features, _logger| value_squared_error_update(actor, features, value_optimizer),
            logger,
        );
    }
}

fn policy_gradient_update<OS, AS, P, V, PO>(
    actor: &PolicyValueNetActor<OS, AS, P, V>,
    features: &LazyPackedHistoryFeatures<OS, AS>,
    policy_optimizer: &PO,
) -> Tensor
where
    OS: FeatureSpace<Tensor>,
    AS: ReprSpace<Tensor> + ParameterizedSampleSpace<Tensor>,
    P: SequenceModule,
    V: StepValue,
    PO: Optimizer,
{
    let step_values = tch::no_grad(|| actor.value.seq_packed(features));

    let policy_output = actor.policy.seq_packed(
        features.observation_features(),
        features.batch_sizes_tensor(),
    );

    let entropies = Cell::new(None);
    let policy_loss_fn = || {
        let (log_probs, entropies_) = actor
            .action_space
            .batch_statistics(&policy_output, features.actions());
        entropies.set(Some(entropies_));
        -(log_probs * &step_values).mean(Kind::Float)
    };

    let _ = policy_optimizer.backward_step(&policy_loss_fn).unwrap();

    entropies.into_inner().unwrap().mean(Kind::Float)
}

fn value_squared_error_update<OS, AS, P, V, VO>(
    actor: &PolicyValueNetActor<OS, AS, P, V>,
    features: &LazyPackedHistoryFeatures<OS, AS>,
    value_optimizer: &VO,
) -> Tensor
where
    OS: FeatureSpace<Tensor>,
    AS: ReprSpace<Tensor>,
    V: StepValue,
    VO: Optimizer,
{
    value_optimizer
        .backward_step(&|| actor.value.loss(features).unwrap())
        .unwrap()
}

#[cfg(test)]
mod gae_policy_gradient {
    use super::*;
    use crate::agents::testing;
    use crate::torch::modules::MlpConfig;
    use crate::torch::optimizers::AdamConfig;
    use crate::torch::seq_modules::{GruMlp, RnnMlpConfig, WithState};
    use crate::torch::step_value::{Gae, GaeConfig, Return};
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
        config.actor_config.steps_per_epoch = 1000;
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
