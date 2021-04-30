//! Recurrent neural network configuration
use super::super::super::ModuleBuilder;
use super::{Gru, Lstm};
use tch::nn::Path;

/// Configuration of a recurrent neural network.
#[derive(Debug, Clone)]
pub struct RnnConfig {
    pub has_biases: bool,
}

impl Default for RnnConfig {
    fn default() -> Self {
        Self { has_biases: true }
    }
}

impl ModuleBuilder<Gru> for RnnConfig {
    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Gru {
        Gru::new(
            vs,
            in_dim,
            out_dim,
            self.has_biases,
            0.0, // dropout
        )
    }
}

impl ModuleBuilder<Lstm> for RnnConfig {
    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Lstm {
        Lstm::new(
            vs,
            in_dim,
            out_dim,
            self.has_biases,
            0.0, // dropout
        )
    }
}

#[cfg(test)]
mod rnn_config {
    use super::*;
    use tch::{nn, Device};

    /// Check that the default GRU Config builds successfully
    #[test]
    fn default_gru_builds() {
        let config = RnnConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: Gru = config.build_module(&vs.root(), 3, 2);
    }

    /// Check that the default LSTM Config builds successfully
    #[test]
    fn default_lstm_builds() {
        let config = RnnConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: Lstm = config.build_module(&vs.root(), 3, 2);
    }
}
