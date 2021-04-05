use super::super::ModuleBuilder;
use std::borrow::Borrow;
use std::marker::PhantomData;
use tch::nn;

/// Configuration of a recurrent neural network.
#[derive(Debug)]
pub struct RnnConfig<R: nn::RNN> {
    pub has_biases: bool,
    pub num_layers: usize,
    type_: PhantomData<R>,
}

impl<R: nn::RNN> Default for RnnConfig<R> {
    fn default() -> Self {
        Self {
            num_layers: 1,
            has_biases: true,
            type_: PhantomData,
        }
    }
}

impl ModuleBuilder for RnnConfig<nn::GRU> {
    type Module = nn::GRU;

    fn build<'a, T: Borrow<nn::Path<'a>>>(
        &self,
        vs: T,
        input_dim: usize,
        output_dim: usize,
    ) -> Self::Module {
        let rnn_config = nn::RNNConfig {
            has_biases: self.has_biases,
            num_layers: self.num_layers as i64,
            dropout: 0.0,
            train: true,
            bidirectional: false,
            batch_first: true,
        };
        nn::gru(vs.borrow(), input_dim as i64, output_dim as i64, rnn_config)
    }
}

impl ModuleBuilder for RnnConfig<nn::LSTM> {
    type Module = nn::LSTM;

    fn build<'a, T: Borrow<nn::Path<'a>>>(
        &self,
        vs: T,
        input_dim: usize,
        output_dim: usize,
    ) -> Self::Module {
        let rnn_config = nn::RNNConfig {
            has_biases: self.has_biases,
            num_layers: self.num_layers as i64,
            dropout: 0.0,
            train: true,
            bidirectional: false,
            batch_first: true,
        };
        nn::lstm(vs.borrow(), input_dim as i64, output_dim as i64, rnn_config)
    }
}

#[cfg(test)]
mod rnn_config {
    use super::*;
    use tch::{nn, Device};

    /// Check that the default GRU Config builds successfully
    #[test]
    fn default_gru_builds() {
        let config: RnnConfig<nn::GRU> = Default::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = config.build(&vs.root(), 3, 2);
    }

    /// Check that the default LSTM Config builds successfully
    #[test]
    fn default_lstm_builds() {
        let config: RnnConfig<nn::LSTM> = Default::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = config.build(&vs.root(), 3, 2);
    }
}
