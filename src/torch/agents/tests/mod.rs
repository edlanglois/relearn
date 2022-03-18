mod policy_gradient;
mod ppo;
mod trpo;

use super::critic::Return;
use super::learning_critic::GradOptConfig;

/// Helper for setting the learning rate in unit tests
trait SetLearningRate {
    fn set_learning_rate(&mut self, rate: f64);
}

impl SetLearningRate for Return {
    // No learning rate to set
    fn set_learning_rate(&mut self, _rate: f64) {}
}

impl<CB> SetLearningRate for GradOptConfig<CB> {
    fn set_learning_rate(&mut self, rate: f64) {
        self.optimizer_config.learning_rate = rate;
    }
}
