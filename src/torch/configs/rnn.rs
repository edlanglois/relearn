use super::super::{seq_modules::SeqModRnn, ModuleBuilder};
use tch::nn;

/// Configuration of a recurrent neural network.
#[derive(Debug, Clone)]
pub struct RnnConfig {
    pub has_biases: bool,
    pub num_layers: usize,
}

impl Default for RnnConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            has_biases: true,
        }
    }
}

impl ModuleBuilder<nn::GRU> for RnnConfig {
    fn build(&self, vs: &nn::Path, input_dim: usize, output_dim: usize) -> nn::GRU {
        let rnn_config = nn::RNNConfig {
            has_biases: self.has_biases,
            num_layers: self.num_layers as i64,
            dropout: 0.0,
            train: true,
            bidirectional: false,
            batch_first: true,
        };
        nn::gru(vs, input_dim as i64, output_dim as i64, rnn_config)
    }
}

impl ModuleBuilder<nn::LSTM> for RnnConfig {
    fn build(&self, vs: &nn::Path, input_dim: usize, output_dim: usize) -> nn::LSTM {
        let rnn_config = nn::RNNConfig {
            has_biases: self.has_biases,
            num_layers: self.num_layers as i64,
            dropout: 0.0,
            train: true,
            bidirectional: false,
            batch_first: true,
        };
        nn::lstm(vs, input_dim as i64, output_dim as i64, rnn_config)
    }
}

impl<R> ModuleBuilder<SeqModRnn<R>> for RnnConfig
where
    RnnConfig: ModuleBuilder<R>,
{
    fn build(&self, vs: &nn::Path, input_dim: usize, output_dim: usize) -> SeqModRnn<R> {
        self.build(vs, input_dim, output_dim).into()
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
        let _: nn::GRU = config.build(&vs.root(), 3, 2);
    }

    /// Check that SeqModRnn<GRU> builds successfully
    #[test]
    fn default_seq_mod_gru_builds() {
        let config = RnnConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: SeqModRnn<nn::GRU> = config.build(&vs.root(), 3, 2);
    }

    /// Check that the default LSTM Config builds successfully
    #[test]
    fn default_lstm_builds() {
        let config = RnnConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _: nn::LSTM = config.build(&vs.root(), 3, 2);
    }
}
