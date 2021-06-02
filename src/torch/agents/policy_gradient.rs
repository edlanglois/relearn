//! Vanilla Policy Gradient
use super::super::history::PackedHistoryFeaturesView;
use super::super::seq_modules::{SequenceModule, StatefulIterativeModule};
use super::super::step_value::{StepValue, StepValueBuilder};
use super::super::{ModuleBuilder, Optimizer, OptimizerBuilder};
use super::actor::{HistoryFeatures, PolicyValueNetActor, PolicyValueNetActorConfig};
use crate::agents::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use crate::logging::Logger;
use crate::spaces::{
    BaseFeatureSpace, BatchFeatureSpace, FeatureSpace, ParameterizedDistributionSpace, ReprSpace,
    Space,
};
use crate::utils::distributions::ArrayDistribution;
use crate::EnvStructure;
use std::cell::Cell;
use std::fmt;
use tch::{kind::Kind, Tensor};

/// Configuration for [`PolicyGradientAgent`]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub actor_config: PolicyValueNetActorConfig<PB, VB>,
    pub policy_optimizer_config: POB,
    pub value_optimizer_config: VOB,
}

impl<PB, POB, VB, VOB> PolicyGradientAgentConfig<PB, POB, VB, VOB> {
    pub const fn new(
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

impl<E, PB, P, POB, PO, VB, V, VOB, VO>
    AgentBuilder<PolicyGradientAgent<E::ObservationSpace, E::ActionSpace, P, PO, V, VO>, E>
    for PolicyGradientAgentConfig<PB, POB, VB, VOB>
where
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: Space + BaseFeatureSpace,
    <E as EnvStructure>::ActionSpace: ParameterizedDistributionSpace<Tensor>,
    PB: ModuleBuilder<P>,
    P: SequenceModule + StatefulIterativeModule,
    POB: OptimizerBuilder<PO>,
    VB: StepValueBuilder<V>,
    V: StepValue,
    VOB: OptimizerBuilder<VO>,
{
    #[allow(clippy::type_complexity)]
    fn build_agent(
        &self,
        env: &E,
        _seed: u64,
    ) -> Result<
        PolicyGradientAgent<E::ObservationSpace, E::ActionSpace, P, PO, V, VO>,
        BuildAgentError,
    > {
        Ok(PolicyGradientAgent::new(
            env,
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

    /// Policy optimizer
    policy_optimizer: PO,

    /// Step value function optimizer.
    value_optimizer: VO,
}

impl<OS, AS, P, PO, V, VO> fmt::Debug for PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: Space + fmt::Debug,
    <OS as Space>::Element: fmt::Debug,
    AS: Space + fmt::Debug,
    <AS as Space>::Element: fmt::Debug,
    P: fmt::Debug,
    PO: fmt::Debug,
    V: fmt::Debug,
    VO: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolicyGradientAgent")
            .field("actor", &self.actor)
            .field("policy_optimizer", &self.policy_optimizer)
            .field("value_optimizer", &self.value_optimizer)
            .finish()
    }
}

impl<OS, AS, P, PO, V, VO> PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: Space + BaseFeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    V: StepValue,
{
    pub fn new<E, PB, VB, POB, VOB>(
        env: &E,
        actor_config: &PolicyValueNetActorConfig<PB, VB>,
        policy_optimizer_config: &POB,
        value_optimizer_config: &VOB,
    ) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        PB: ModuleBuilder<P>,
        VB: StepValueBuilder<V>,
        POB: OptimizerBuilder<PO>,
        VOB: OptimizerBuilder<VO>,
    {
        let actor = actor_config.build_actor(env);

        let policy_optimizer = policy_optimizer_config
            .build_optimizer(&actor.policy_variables)
            .unwrap();
        let value_optimizer = value_optimizer_config
            .build_optimizer(&actor.value_variables)
            .unwrap();

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
    AS: ParameterizedDistributionSpace<Tensor>,
    P: StatefulIterativeModule,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        self.actor.act(observation, new_episode)
    }
}

impl<OS, AS, P, PO, V, VO> Agent<OS::Element, AS::Element>
    for PolicyGradientAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor> + BatchFeatureSpace<Tensor>,
    AS: ReprSpace<Tensor> + ParameterizedDistributionSpace<Tensor>,
    P: SequenceModule + StatefulIterativeModule,
    PO: Optimizer,
    V: StepValue,
    VO: Optimizer,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        Actor::act(self, observation, new_episode)
    }

    fn update(&mut self, step: Step<OS::Element, AS::Element>, logger: &mut dyn Logger) {
        let policy_optimizer = &mut self.policy_optimizer;
        let value_optimizer = &mut self.value_optimizer;
        self.actor.update(
            step,
            |actor, features, _logger| policy_gradient_update(actor, features, policy_optimizer),
            |actor, features, _logger| value_squared_error_update(actor, features, value_optimizer),
            logger,
        );
    }
}

/// Perform a single policy gradient update step using the given history features.
pub fn policy_gradient_update<OS, AS, P, V, PO>(
    actor: &PolicyValueNetActor<OS, AS, P, V>,
    features: &HistoryFeatures<OS, AS>,
    optimizer: &mut PO,
) -> Tensor
where
    OS: BatchFeatureSpace<Tensor>,
    AS: ReprSpace<Tensor> + ParameterizedDistributionSpace<Tensor>,
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
        let action_distributions = actor.action_space.distribution(&policy_output);
        let log_probs = action_distributions.log_probs(features.actions());
        entropies.set(Some(action_distributions.entropy()));
        -(log_probs * &step_values).mean(Kind::Float)
    };

    let _ = optimizer.backward_step(&policy_loss_fn).unwrap();

    entropies.into_inner().unwrap().mean(Kind::Float)
}

/// Perform a single squared error loss value function update using the given history features.
pub fn value_squared_error_update<OS, AS, P, V, VO>(
    actor: &PolicyValueNetActor<OS, AS, P, V>,
    features: &HistoryFeatures<OS, AS>,
    optimizer: &mut VO,
) -> Tensor
where
    OS: BatchFeatureSpace<Tensor>,
    AS: ReprSpace<Tensor>,
    V: StepValue,
    VO: Optimizer,
{
    optimizer
        .backward_step(&|| actor.value.loss(features).unwrap())
        .unwrap()
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod policy_gradient {
    use super::*;
    use crate::agents::testing;
    use crate::torch::modules::MlpConfig;
    use crate::torch::optimizers::AdamConfig;
    use crate::torch::seq_modules::{GruMlp, RnnMlpConfig, WithState};
    use crate::torch::step_value::{Gae, GaeConfig, Return};
    use tch::{nn::Sequential, Device};

    fn test_train_policy_gradient<P, PB, V, VB>(
        mut config: PolicyGradientAgentConfig<PB, AdamConfig, VB, AdamConfig>,
    ) where
        P: SequenceModule + StatefulIterativeModule,
        PB: ModuleBuilder<P> + Default,
        V: StepValue,
        VB: StepValueBuilder<V> + Default,
    {
        // Speed up learning for this simple environment
        config.actor_config.steps_per_epoch = 25;
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
        test_train_policy_gradient::<Sequential, MlpConfig, Return, Return>(Default::default())
    }

    #[test]
    fn default_mlp_return_learns_derministic_bandit_cuda_if_available() {
        let mut config = PolicyGradientAgentConfig::default();
        config.actor_config.device = Device::cuda_if_available();
        test_train_policy_gradient::<Sequential, MlpConfig, Return, Return>(config)
    }

    #[test]
    fn default_mlp_gae_mlp_learns_derministic_bandit() {
        test_train_policy_gradient::<Sequential, MlpConfig, Gae<Sequential>, GaeConfig<MlpConfig>>(
            Default::default(),
        )
    }

    #[test]
    fn default_gru_mlp_return_learns_derministic_bandit() {
        test_train_policy_gradient::<WithState<GruMlp>, RnnMlpConfig, Return, Return>(
            Default::default(),
        )
    }

    #[test]
    fn default_gru_mlp_gae_mlp_derministic_bandit() {
        test_train_policy_gradient::<
            WithState<GruMlp>,
            RnnMlpConfig,
            Gae<Sequential>,
            GaeConfig<MlpConfig>,
        >(Default::default())
    }

    #[test]
    fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
        test_train_policy_gradient::<
            WithState<GruMlp>,
            RnnMlpConfig,
            Gae<WithState<GruMlp>>,
            GaeConfig<RnnMlpConfig>,
        >(Default::default())
    }
}
