//! Vanilla DQN agent tests.
use crate::agents::testing;
use crate::torch::agents::{dqn::DqnConfig, schedules::DataCollectionSchedule};
use crate::torch::modules::{
    BuildModule, GruMlpConfig, IterativeModule, MlpConfig, SequenceModule,
};
use crate::torch::optimizers::AdamConfig;
use tch::Device;

fn test_train_policy_gradient<VB>(mut config: DqnConfig<VB, AdamConfig>)
where
    VB: BuildModule,
    VB::Module: IterativeModule + SequenceModule,
{
    // Speed up learning for this simple environment
    config.optimizer_config.learning_rate = 0.1;
    config.minibatch_steps = 10;
    config.update_size = DataCollectionSchedule::Constant(10);
    testing::train_deterministic_bandit(&config, 10, 0.9);
}

#[test]
fn default_mlp_learns_derministic_bandit() {
    test_train_policy_gradient::<MlpConfig>(Default::default())
}

#[test]
fn default_mlp_learns_derministic_bandit_cuda_if_available() {
    let config = DqnConfig {
        device: Device::cuda_if_available(),
        ..DqnConfig::default()
    };
    test_train_policy_gradient::<MlpConfig>(config)
}

#[test]
fn default_gru_mlp_learns_derministic_bandit() {
    test_train_policy_gradient::<GruMlpConfig>(Default::default())
}

#[test]
fn default_gru_mlp_learns_derministic_bandit_cuda_if_available() {
    let config = DqnConfig {
        device: Device::cuda_if_available(),
        ..DqnConfig::default()
    };
    test_train_policy_gradient::<GruMlpConfig>(config)
}
