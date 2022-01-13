//! Policy-gradient actor critic agent tests.
use crate::agents::testing;
use crate::torch::{
    agents::ActorCriticConfig,
    critic::{BuildCritic, GaeConfig, Return},
    modules::MlpConfig,
    optimizers::{AdamConfig, ConjugateGradientOptimizerConfig},
    policy::BuildPolicy,
    seq_modules::{GruMlpConfig, IterativeModule},
    updaters::{CriticLossUpdateRule, TrpoPolicyUpdateRule, WithOptimizer},
};
use tch::Device;

fn test_train_trpo<PB, CB>(
    mut config: ActorCriticConfig<
        PB,
        WithOptimizer<TrpoPolicyUpdateRule, ConjugateGradientOptimizerConfig>,
        CB,
        WithOptimizer<CriticLossUpdateRule, AdamConfig>,
    >,
) where
    PB: BuildPolicy + Clone,
    PB::Policy: IterativeModule,
    CB: BuildCritic,
{
    // Speed up learning for this simple environment
    config.critic_updater_config.optimizer.learning_rate = 0.1;
    config.min_batch_steps = 25;
    testing::train_deterministic_bandit(&config, 10, 0.9);
}

#[test]
fn default_mlp_return_learns_derministic_bandit() {
    test_train_trpo::<MlpConfig, Return>(Default::default())
}

#[test]
fn default_mlp_return_learns_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_trpo::<MlpConfig, Return>(config)
}

#[test]
fn default_mlp_gae_mlp_learns_derministic_bandit() {
    test_train_trpo::<MlpConfig, GaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_return_learns_derministic_bandit() {
    test_train_trpo::<GruMlpConfig, Return>(Default::default())
}

#[test]
fn default_gru_mlp_gae_mlp_derministic_bandit() {
    test_train_trpo::<GruMlpConfig, GaeConfig<MlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit() {
    test_train_trpo::<GruMlpConfig, GaeConfig<GruMlpConfig>>(Default::default())
}

#[test]
fn default_gru_mlp_gae_gru_mlp_derministic_bandit_cuda_if_available() {
    let config = ActorCriticConfig {
        device: Device::cuda_if_available(),
        ..ActorCriticConfig::default()
    };
    test_train_trpo::<GruMlpConfig, GaeConfig<GruMlpConfig>>(config)
}
