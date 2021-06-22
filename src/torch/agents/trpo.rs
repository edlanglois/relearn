//! Trust Region Policy Optimization
//!
//! # Reference
//! Schulman, John, et al. "Trust region policy optimization."
//! International conference on machine learning. PMLR, 2015.
//! <https://arxiv.org/abs/1502.05477>

use super::super::critic::{Critic, CriticBuilder};
use super::super::history::PackedHistoryFeaturesView;
use super::super::optimizers::{
    Optimizer, OptimizerBuilder, OptimizerStepError, TrustRegionOptimizer,
};
use super::super::seq_modules::{SequenceModule, StatefulIterativeModule};
use super::super::ModuleBuilder;
use super::actor::{HistoryFeatures, PolicyValueNetActor, PolicyValueNetActorConfig};
use super::policy_gradient;
use crate::agents::{Actor, Agent, AgentBuilder, BuildAgentError, Step};
use crate::logging::{Logger, TimeSeriesEventLogger, TimeSeriesLogger};
use crate::spaces::{
    BaseFeatureSpace, BatchFeatureSpace, FeatureSpace, ParameterizedDistributionSpace, ReprSpace,
    Space,
};
use crate::torch::backends::{CudnnSupport, WithCudnnEnabled};
use crate::utils::distributions::ArrayDistribution;
use crate::EnvStructure;
use std::fmt;
use tch::{kind::Kind, Tensor};

/// Configuration for [`TrpoAgent`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrpoAgentConfig<PB, POB, VB, VOB> {
    pub actor_config: PolicyValueNetActorConfig<PB, VB>,
    pub policy_optimizer_config: POB,
    pub value_optimizer_config: VOB,
    /// Maximum policy KL divergence when taking a step.
    pub max_policy_step_kl: f64,
}

impl<PB, POB, VB, VOB> TrpoAgentConfig<PB, POB, VB, VOB> {
    pub const fn new(
        actor_config: PolicyValueNetActorConfig<PB, VB>,
        policy_optimizer_config: POB,
        value_optimizer_config: VOB,
        max_policy_step_kl: f64,
    ) -> Self {
        Self {
            actor_config,
            policy_optimizer_config,
            value_optimizer_config,
            max_policy_step_kl,
        }
    }
}

impl<PB, POB, VB, VOB> Default for TrpoAgentConfig<PB, POB, VB, VOB>
where
    PolicyValueNetActorConfig<PB, VB>: Default,
    POB: Default,
    VOB: Default,
{
    fn default() -> Self {
        Self {
            actor_config: PolicyValueNetActorConfig::default(),
            policy_optimizer_config: POB::default(),
            value_optimizer_config: VOB::default(),
            max_policy_step_kl: 0.01,
        }
    }
}

impl<E, PB, P, POB, PO, VB, V, VOB, VO>
    AgentBuilder<TrpoAgent<E::ObservationSpace, E::ActionSpace, P, PO, V, VO>, E>
    for TrpoAgentConfig<PB, POB, VB, VOB>
where
    E: EnvStructure + ?Sized,
    <E as EnvStructure>::ObservationSpace: Space + BaseFeatureSpace,
    <E as EnvStructure>::ActionSpace: ParameterizedDistributionSpace<Tensor>,
    PB: ModuleBuilder<P>,
    P: SequenceModule + StatefulIterativeModule,
    POB: OptimizerBuilder<PO>,
    VB: CriticBuilder<V>,
    V: Critic,
    VOB: OptimizerBuilder<VO>,
{
    #[allow(clippy::type_complexity)]
    fn build_agent(
        &self,
        env: &E,
        _seed: u64,
    ) -> Result<TrpoAgent<E::ObservationSpace, E::ActionSpace, P, PO, V, VO>, BuildAgentError> {
        Ok(TrpoAgent::new(
            env,
            &self.actor_config,
            &self.policy_optimizer_config,
            &self.value_optimizer_config,
            self.max_policy_step_kl,
        ))
    }
}

/// Trust Region Policy Optimization Agent
pub struct TrpoAgent<OS, AS, P, PO, V, VO>
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

    /// Maximum policy KL divergence when taking a step.
    max_policy_step_kl: f64,
}

impl<OS, AS, P, PO, V, VO> fmt::Debug for TrpoAgent<OS, AS, P, PO, V, VO>
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
        f.debug_struct("TrpoAgent")
            .field("actor", &self.actor)
            .field("policy_optimizer", &self.policy_optimizer)
            .field("value_optimizer", &self.value_optimizer)
            .field("max_policy_step_kl", &self.max_policy_step_kl)
            .finish()
    }
}

impl<OS, AS, P, PO, V, VO> TrpoAgent<OS, AS, P, PO, V, VO>
where
    OS: Space + BaseFeatureSpace,
    AS: ParameterizedDistributionSpace<Tensor>,
    V: Critic,
{
    pub fn new<E, PB, VB, POB, VOB>(
        env: &E,
        actor_config: &PolicyValueNetActorConfig<PB, VB>,
        policy_optimizer_config: &POB,
        value_optimizer_config: &VOB,
        max_policy_step_kl: f64,
    ) -> Self
    where
        E: EnvStructure<ObservationSpace = OS, ActionSpace = AS> + ?Sized,
        PB: ModuleBuilder<P>,
        VB: CriticBuilder<V>,
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
            max_policy_step_kl,
        }
    }
}

impl<OS, AS, P, PO, V, VO> Actor<OS::Element, AS::Element> for TrpoAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor>,
    AS: ParameterizedDistributionSpace<Tensor>,
    P: StatefulIterativeModule,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        self.actor.act(observation, new_episode)
    }
}

impl<OS, AS, P, PO, V, VO> Agent<OS::Element, AS::Element> for TrpoAgent<OS, AS, P, PO, V, VO>
where
    OS: FeatureSpace<Tensor> + BatchFeatureSpace<Tensor>,
    AS: ReprSpace<Tensor> + ParameterizedDistributionSpace<Tensor>,
    P: SequenceModule + StatefulIterativeModule + CudnnSupport,
    PO: TrustRegionOptimizer,
    V: Critic,
    VO: Optimizer,
{
    fn act(&mut self, observation: &OS::Element, new_episode: bool) -> AS::Element {
        Actor::act(self, observation, new_episode)
    }

    fn update(&mut self, step: Step<OS::Element, AS::Element>, logger: &mut dyn TimeSeriesLogger) {
        let policy_optimizer = &mut self.policy_optimizer;
        let value_optimizer = &mut self.value_optimizer;
        let max_policy_step_kl = self.max_policy_step_kl;
        self.actor.update(
            step,
            |actor, features, logger| {
                trpo_update(
                    actor,
                    features,
                    policy_optimizer,
                    max_policy_step_kl,
                    logger,
                )
            },
            |actor, features, logger| {
                policy_gradient::value_squared_error_update(
                    actor,
                    features,
                    value_optimizer,
                    logger,
                )
            },
            logger,
        );
    }
}

fn trpo_update<OS, AS, P, PO, V>(
    actor: &PolicyValueNetActor<OS, AS, P, V>,
    features: &HistoryFeatures<OS, AS>,
    policy_optimizer: &mut PO,
    max_policy_step_kl: f64,
    logger: &mut TimeSeriesEventLogger,
) -> Option<Tensor>
where
    OS: BatchFeatureSpace<Tensor>,
    AS: ReprSpace<Tensor> + ParameterizedDistributionSpace<Tensor>,
    P: SequenceModule + CudnnSupport,
    PO: TrustRegionOptimizer,
    V: Critic,
{
    let mut logger = logger.scope("trpo");
    if features.episode_ranges().is_empty() {
        logger
            .log(
                "no_policy_step",
                "Skipping policy update: empty history features".into(),
            )
            .unwrap();
        return None;
    }

    let _cudnn_disable_guard = if actor.policy.has_cudnn_second_derivatives() {
        None
    } else {
        Some(WithCudnnEnabled::new(false))
    };

    let observation_features = features.observation_features();
    let batch_sizes = features.batch_sizes_tensor();
    let actions = features.actions();

    let (critics, initial_distribution, initial_log_probs, initial_policy_entropy) = {
        let _no_grad = tch::no_grad_guard();

        let critics = actor.value.seq_packed(features);
        let policy_output = actor.policy.seq_packed(observation_features, batch_sizes);
        let distribution = actor.action_space.distribution(&policy_output);
        let log_probs = distribution.log_probs(actions);
        let entropy = distribution.entropy().mean(Kind::Float);

        (critics, distribution, log_probs, entropy)
    };

    let policy_loss_distance_fn = || {
        let policy_output = actor.policy.seq_packed(observation_features, batch_sizes);
        let distribution = actor.action_space.distribution(&policy_output);

        let log_probs = distribution.log_probs(actions);
        let likelihood_ratio = (log_probs - &initial_log_probs).exp();
        let loss = -(likelihood_ratio * &critics).mean(Kind::Float);

        // NOTE:
        // The [TRPO paper] and [Garage] use `KL(old_policy || new_policy)` while
        // [Spinning Up] uses `KL(new_policy || old_policy)`.
        //
        // I do not know why Spinning Up differs. I follow the TRPO paper and Garage.
        //
        // [TRPO paper]: <https://arxiv.org/abs/1502.05477>
        // [Garage]: <https://garage.readthedocs.io/en/latest/user/algo_trpo.html>
        // [Spinning Up]: <https://spinningup.openai.com/en/latest/algorithms/trpo.html>
        let distance = initial_distribution
            .kl_divergence_from(&distribution)
            .mean(Kind::Float);

        (loss, distance)
    };

    let result = policy_optimizer.trust_region_backward_step(
        &policy_loss_distance_fn,
        max_policy_step_kl,
        &mut logger,
    );
    if let Err(error) = result {
        match error {
            OptimizerStepError::NaNLoss => panic!("NaN loss in policy optimization"),
            OptimizerStepError::NaNConstraint => panic!("NaN constraint in policy optimization"),
            e => logger.log("no_policy_step", e.to_string().into()).unwrap(),
        }
    }

    Some(initial_policy_entropy)
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod trpo {
    use super::*;
    use crate::agents::testing;
    use crate::torch::critic::{Gae, GaeConfig, Return};
    use crate::torch::modules::MlpConfig;
    use crate::torch::optimizers::{AdamConfig, ConjugateGradientOptimizerConfig};
    use crate::torch::seq_modules::{GruMlp, RnnMlpConfig, WithState};
    use tch::{nn::Sequential, Device};

    fn test_train_default_trpo<P, PB, V, VB>(
        mut config: TrpoAgentConfig<PB, ConjugateGradientOptimizerConfig, VB, AdamConfig>,
    ) where
        P: SequenceModule + StatefulIterativeModule + CudnnSupport,
        PB: ModuleBuilder<P> + Default,
        V: Critic,
        VB: CriticBuilder<V> + Default,
    {
        // Speed up learning for this simple environment
        config.actor_config.steps_per_epoch = 25;
        config.value_optimizer_config.learning_rate = 0.1;
        testing::train_deterministic_bandit(
            |env_structure| -> TrpoAgent<_, _, P, _, V, _> {
                config.build_agent(env_structure, 0).unwrap()
            },
            1_000,
            0.9,
        );
    }

    #[test]
    fn default_mlp_return_learns_derministic_bandit() {
        test_train_default_trpo::<Sequential, MlpConfig, Return, Return>(Default::default());
    }

    #[test]
    fn default_mlp_return_learns_derministic_bandit_cuda_if_available() {
        let mut config = TrpoAgentConfig::default();
        config.actor_config.device = Device::cuda_if_available();
        test_train_default_trpo::<Sequential, MlpConfig, Return, Return>(config);
    }

    #[test]
    fn default_mlp_gae_mlp_learns_derministic_bandit() {
        test_train_default_trpo::<Sequential, MlpConfig, Gae<Sequential>, GaeConfig<MlpConfig>>(
            Default::default(),
        )
    }

    #[test]
    fn default_gru_mlp_return_learns_derministic_bandit() {
        test_train_default_trpo::<WithState<GruMlp>, RnnMlpConfig, Return, Return>(
            Default::default(),
        )
    }

    #[test]
    fn default_gru_mlp_gae_mlp_derministic_bandit() {
        test_train_default_trpo::<
            WithState<GruMlp>,
            RnnMlpConfig,
            Gae<Sequential>,
            GaeConfig<MlpConfig>,
        >(Default::default())
    }

    #[test]
    fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
        test_train_default_trpo::<
            WithState<GruMlp>,
            RnnMlpConfig,
            Gae<WithState<GruMlp>>,
            GaeConfig<RnnMlpConfig>,
        >(Default::default())
    }

    #[test]
    fn default_gru_mlp_gae_gru_mlp_derministic_bandit_cuda_if_available() {
        let mut config = TrpoAgentConfig::default();
        config.actor_config.device = Device::cuda_if_available();
        test_train_default_trpo::<
            WithState<GruMlp>,
            RnnMlpConfig,
            Gae<WithState<GruMlp>>,
            GaeConfig<RnnMlpConfig>,
        >(config)
    }
}
