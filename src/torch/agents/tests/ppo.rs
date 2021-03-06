//! Policy-gradient actor critic agent tests.
use super::SetLearningRate;
use crate::agents::testing;
use crate::torch::agents::{
    critic::{GaeConfig, Return},
    learning_critic::{BuildLearningCritic, GradOptConfig},
    learning_policy::PpoConfig,
    policy::BuildPolicy,
    ActorCriticConfig,
};
use crate::torch::modules::{GruMlpConfig, MlpConfig};
use tch::Device;

fn test_train_ppo<PB, LCB>(mut config: ActorCriticConfig<PpoConfig<PB>, LCB>)
where
    PB: BuildPolicy,
    LCB: BuildLearningCritic + SetLearningRate,
{
    // Speed up learning for this simple environment
    config.policy_config.optimizer_config.learning_rate = 0.1;
    config.critic_config.set_learning_rate(0.1);
    config.min_batch_steps = 25;
    testing::train_deterministic_bandit(&config, 10, 0.9);
}

type OptGaeConfig<T> = GradOptConfig<GaeConfig<T>>;

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
    test_train_ppo::<MlpConfig, OptGaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_return_learns_derministic_bandit() {
    test_train_ppo::<GruMlpConfig, Return>(Default::default())
}

#[test]
fn default_gru_mlp_gae_mlp_derministic_bandit() {
    test_train_ppo::<GruMlpConfig, OptGaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
    test_train_ppo::<GruMlpConfig, OptGaeConfig<GruMlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_ppo::<GruMlpConfig, OptGaeConfig<GruMlpConfig>>(config)
}
