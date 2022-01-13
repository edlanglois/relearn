//! Policy-gradient actor critic agent tests.
use crate::agents::testing;
use crate::torch::{
    agents::ActorCriticConfig,
    critic::{BuildCritic, GaeConfig, Return},
    modules::MlpConfig,
    optimizers::AdamConfig,
    policy::BuildPolicy,
    seq_modules::{GruMlpConfig, IterativeModule},
    updaters::{CriticLossUpdateRule, PpoPolicyUpdateRule, WithOptimizer},
};
use tch::Device;

fn test_train_ppo<PB, CB>(
    mut config: ActorCriticConfig<
        PB,
        WithOptimizer<PpoPolicyUpdateRule, AdamConfig>,
        CB,
        WithOptimizer<CriticLossUpdateRule, AdamConfig>,
    >,
) where
    PB: BuildPolicy + Clone,
    PB::Policy: IterativeModule,
    CB: BuildCritic,
{
    // Speed up learning for this simple environment
    config.policy_updater_config.optimizer.learning_rate = 0.1;
    config.critic_updater_config.optimizer.learning_rate = 0.1;
    config.min_batch_steps = 25;
    testing::train_deterministic_bandit(&config, 10, 0.9);
}

#[test]
fn default_mlp_return_learns_derministic_bandit() {
    test_train_ppo::<MlpConfig, Return>(Default::default())
}

#[test]
fn default_mlp_return_learns_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_ppo::<MlpConfig, Return>(config)
}

#[test]
fn default_mlp_gae_mlp_learns_derministic_bandit() {
    test_train_ppo::<MlpConfig, GaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_return_learns_derministic_bandit() {
    test_train_ppo::<GruMlpConfig, Return>(Default::default())
}

#[test]
fn default_gru_mlp_gae_mlp_derministic_bandit() {
    test_train_ppo::<GruMlpConfig, GaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
    test_train_ppo::<GruMlpConfig, GaeConfig<GruMlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_ppo::<GruMlpConfig, GaeConfig<GruMlpConfig>>(config)
}
