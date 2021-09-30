//! PPO actor-critic tests
use crate::agents::{testing, BuildAgent};
use crate::torch::agents::ActorCriticConfig;
use crate::torch::critic::{BuildCritic, GaeConfig, Return};
use crate::torch::modules::{BuildModule, MlpConfig};
use crate::torch::optimizers::AdamConfig;
use crate::torch::policy::Policy;
use crate::torch::seq_modules::{GruMlpConfig, WithStateConfig};
use crate::torch::updaters::{CriticLossUpdateRule, PpoPolicyUpdateRule, WithOptimizer};
use tch::Device;

fn test_train_policy_gradient<PB, CB>(
    mut config: ActorCriticConfig<
        PB,
        WithOptimizer<PpoPolicyUpdateRule, AdamConfig>,
        CB,
        WithOptimizer<CriticLossUpdateRule, AdamConfig>,
    >,
) where
    PB: BuildModule,
    <PB as BuildModule>::Module: Policy,
    CB: BuildCritic,
{
    // Speed up learning for this simple environment
    config.steps_per_epoch = 25;
    config.policy_updater_config.optimizer.learning_rate = 0.1;
    config.critic_updater_config.optimizer.learning_rate = 0.1;
    testing::train_deterministic_bandit(|env| config.build_agent(env, 0).unwrap(), 1_000, 0.9);
}

#[test]
fn default_mlp_return_learns_derministic_bandit() {
    test_train_policy_gradient::<MlpConfig, Return>(Default::default())
}

#[test]
fn default_mlp_return_learns_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_policy_gradient::<MlpConfig, Return>(config)
}

#[test]
fn default_mlp_gae_mlp_learns_derministic_bandit() {
    test_train_policy_gradient::<MlpConfig, GaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_return_learns_derministic_bandit() {
    test_train_policy_gradient::<WithStateConfig<GruMlpConfig>, Return>(Default::default())
}

#[test]
fn default_gru_mlp_gae_mlp_derministic_bandit() {
    test_train_policy_gradient::<WithStateConfig<GruMlpConfig>, GaeConfig<MlpConfig>>(
        Default::default(),
    )
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
    test_train_policy_gradient::<WithStateConfig<GruMlpConfig>, GaeConfig<GruMlpConfig>>(
        Default::default(),
    )
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_policy_gradient::<WithStateConfig<GruMlpConfig>, GaeConfig<GruMlpConfig>>(config)
}
