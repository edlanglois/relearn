//! Policy-gradient actor critic agent tests.
use super::SetLearningRate;
use crate::agents::testing;
use crate::torch::agents::{
    critic::{GaeConfig, RewardToGo},
    learning_critic::{BuildLearningCritic, GradOptConfig},
    learning_policy::TrpoConfig,
    policy::BuildPolicy,
    ActorCriticConfig,
};
use crate::torch::modules::{GruMlpConfig, MlpConfig, Module};
use tch::Device;

fn test_train_trpo<PB, LCB>(mut config: ActorCriticConfig<TrpoConfig<PB>, LCB>)
where
    PB: BuildPolicy,
    PB::Policy: Module,
    LCB: BuildLearningCritic + SetLearningRate,
{
    // Speed up learning for this simple environment
    config.critic_config.set_learning_rate(0.1);
    config.min_batch_steps = 25;
    testing::train_deterministic_bandit(&config, 10, 0.9);
}

type OptGaeConfig<T> = GradOptConfig<GaeConfig<T>>;

#[test]
fn default_mlp_return_learns_derministic_bandit() {
    test_train_trpo::<MlpConfig, RewardToGo>(Default::default())
}

#[test]
fn default_mlp_return_learns_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_trpo::<MlpConfig, RewardToGo>(config)
}

#[test]
fn default_mlp_gae_mlp_learns_derministic_bandit() {
    test_train_trpo::<MlpConfig, OptGaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_return_learns_derministic_bandit() {
    test_train_trpo::<GruMlpConfig, RewardToGo>(Default::default())
}

#[test]
fn default_gru_mlp_gae_mlp_derministic_bandit() {
    test_train_trpo::<GruMlpConfig, OptGaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
    test_train_trpo::<GruMlpConfig, OptGaeConfig<GruMlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_trpo::<GruMlpConfig, OptGaeConfig<GruMlpConfig>>(config)
}
