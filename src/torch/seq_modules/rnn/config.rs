//! Recurrent neural network configuration
use super::super::super::BuildModule;
use super::{Gru, Lstm};
use tch::nn::Path;

macro_rules! rnn_config {
    ($config:ident, $module:ty, $name:expr) => {
        #[doc = concat!("Configuration of a [", $name, "](", stringify!($module), ") recurrent neural network")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $config {
            pub has_biases: bool,
        }

        impl Default for $config {
            fn default() -> Self {
                Self { has_biases: true }
            }
        }

        impl BuildModule for $config {
            type Module = $module;

            fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
                <$module>::new(
                    vs,
                    in_dim,
                    out_dim,
                    self.has_biases,
                    0.0, // dropout
                )
            }
        }
    };
}

rnn_config!(GruConfig, Gru, "GRU");
rnn_config!(LstmConfig, Lstm, "LSTM");

#[cfg(test)]
mod rnn_config {
    use super::*;
    use tch::{nn, Device};

    /// Check that the default GRU Config builds successfully
    #[test]
    fn default_gru_builds() {
        let config = GruConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = config.build_module(&vs.root(), 3, 2);
    }

    /// Check that the default LSTM Config builds successfully
    #[test]
    fn default_lstm_builds() {
        let config = LstmConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = config.build_module(&vs.root(), 3, 2);
    }
}
