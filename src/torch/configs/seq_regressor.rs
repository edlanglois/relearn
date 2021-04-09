use super::super::seq_modules::{SeqModRnn, SequenceRegressor};
use super::super::{Activation, ModuleBuilder};
use super::{MlpConfig, RnnConfig};
use std::borrow::Borrow;
use tch::nn;

/// Configuration for an RNN followed by a feed-forward network.
#[derive(Debug, Clone)]
pub struct SequenceRegressorConfig<R, P> {
    pub rnn_config: R,
    pub rnn_hidden_size: usize,
    pub rnn_output_activation: Activation,
    pub post_config: P,
}

/// Configuration for an MLP stacked on top of a GRU.
pub type GruMlpConfig = SequenceRegressorConfig<RnnConfig<nn::GRU>, MlpConfig>;
/// Configuration for an MLP stacked on top of an LSTM.
pub type LstmMlpConfig = SequenceRegressorConfig<RnnConfig<nn::LSTM>, MlpConfig>;

impl<R, P> Default for SequenceRegressorConfig<R, P>
where
    R: Default,
    P: Default,
{
    fn default() -> Self {
        Self {
            rnn_config: Default::default(),
            rnn_hidden_size: 128,
            rnn_output_activation: Activation::Relu,
            post_config: Default::default(),
        }
    }
}

impl<R, P> ModuleBuilder for SequenceRegressorConfig<R, P>
where
    R: ModuleBuilder,
    <R as ModuleBuilder>::Module: nn::RNN,
    P: ModuleBuilder,
    <P as ModuleBuilder>::Module: nn::Module,
{
    type Module = SequenceRegressor<
        'static,
        SeqModRnn<<R as ModuleBuilder>::Module>,
        <P as ModuleBuilder>::Module,
    >;

    fn build<'a, T: Borrow<nn::Path<'a>>>(
        &self,
        vs: T,
        input_dim: usize,
        output_dim: usize,
    ) -> Self::Module {
        let vs = vs.borrow();
        SequenceRegressor::new(
            self.rnn_config
                .build(vs / "rnn", input_dim, self.rnn_hidden_size)
                .into(),
            self.rnn_output_activation.maybe_module(),
            self.post_config
                .build(vs / "post", self.rnn_hidden_size, output_dim),
        )
    }
}

#[cfg(test)]
mod sequence_regressor_config {
    use super::super::super::seq_modules::testing;
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    type GruMlp = <GruMlpConfig as ModuleBuilder>::Module;

    #[fixture]
    fn gru_mlp_default_module() -> (GruMlp, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = GruMlpConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let module = config.build(&vs.root(), in_dim, out_dim);
        (module, in_dim, out_dim)
    }

    #[rstest]
    fn gru_mlp_default_module_seq_serial(gru_mlp_default_module: (GruMlp, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp_default_module;
        testing::check_seq_serial(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_default_module_step(gru_mlp_default_module: (GruMlp, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp_default_module;
        testing::check_step(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_default_module_iter_map(gru_mlp_default_module: (GruMlp, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp_default_module;
        testing::check_iter_map(&gru_mlp, in_dim, out_dim);
    }

    type LstmMlp = <LstmMlpConfig as ModuleBuilder>::Module;

    #[fixture]
    fn lstm_mlp_default_module() -> (LstmMlp, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = LstmMlpConfig::default();
        let vs = nn::VarStore::new(Device::Cpu);
        let module = config.build(&vs.root(), in_dim, out_dim);
        (module, in_dim, out_dim)
    }

    #[rstest]
    fn lstm_mlp_default_module_seq_serial(lstm_mlp_default_module: (LstmMlp, usize, usize)) {
        let (lstm_mlp, in_dim, out_dim) = lstm_mlp_default_module;
        testing::check_seq_serial(&lstm_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn lstm_mlp_default_module_step(lstm_mlp_default_module: (LstmMlp, usize, usize)) {
        let (lstm_mlp, in_dim, out_dim) = lstm_mlp_default_module;
        testing::check_step(&lstm_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn lstm_mlp_default_module_iter_map(lstm_mlp_default_module: (LstmMlp, usize, usize)) {
        let (lstm_mlp, in_dim, out_dim) = lstm_mlp_default_module;
        testing::check_iter_map(&lstm_mlp, in_dim, out_dim);
    }
}
