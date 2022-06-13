use super::{
    reward_to_go, BuildCritic, Critic, Device, HistoryFeatures, PackedTensor, StatsLogger,
};
use serde::{Deserialize, Serialize};

/// Configuration for [`RewardToGo`]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RewardToGoConfig;

impl BuildCritic for RewardToGoConfig {
    type Critic = RewardToGo;

    fn build_critic(&self, _in_dim: usize, discount_factor: f64, _device: Device) -> Self::Critic {
        #[allow(clippy::cast_possible_truncation)]
        RewardToGo {
            discount_factor: discount_factor as f32,
        }
    }
}

/// Reward-to-go critic. Estimates action values as the discounted sum of future rewards.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardToGo {
    discount_factor: f32,
}

impl Critic for RewardToGo {
    fn advantages(&self, features: &dyn HistoryFeatures) -> PackedTensor {
        reward_to_go(self.discount_factor, features)
    }

    fn update(&mut self, _features: &dyn HistoryFeatures, _logger: &mut dyn StatsLogger) {}
}
