//! PPO actor-critic tests
use crate::agents::{testing, AgentBuilder};
use crate::torch::agents::{ActorCriticAgent, ActorCriticConfig};
use crate::torch::critic::{Critic, CriticBuilder, Gae, GaeConfig, Return};
use crate::torch::modules::{MlpConfig, ModuleBuilder};
use crate::torch::optimizers::AdamConfig;
use crate::torch::seq_modules::{
    GruMlp, RnnMlpConfig, SequenceModule, StatefulIterativeModule, WithState,
};
use crate::torch::updaters::{CriticLossUpdateRule, PpoPolicyUpdateRule, WithOptimizer};
use tch::{nn::Sequential, Device};

fn test_train_policy_gradient<P, PB, V, VB>(
    mut config: ActorCriticConfig<
        PB,
        WithOptimizer<PpoPolicyUpdateRule, AdamConfig>,
        VB,
        WithOptimizer<CriticLossUpdateRule, AdamConfig>,
    >,
) where
    P: SequenceModule + StatefulIterativeModule,
    PB: ModuleBuilder<P>,
    V: Critic,
    VB: CriticBuilder<V>,
{
    // Speed up learning for this simple environment
    config.steps_per_epoch = 25;
    config.policy_updater_config.optimizer.learning_rate = 0.1;
    config.critic_updater_config.optimizer.learning_rate = 0.1;
    testing::train_deterministic_bandit(
        |env_structure| -> ActorCriticAgent<_, _, P, _, V, _> {
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
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
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
    test_train_policy_gradient::<WithState<GruMlp>, RnnMlpConfig, Return, Return>(Default::default())
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

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_policy_gradient::<
        WithState<GruMlp>,
        RnnMlpConfig,
        Gae<WithState<GruMlp>>,
        GaeConfig<RnnMlpConfig>,
    >(config)
}
