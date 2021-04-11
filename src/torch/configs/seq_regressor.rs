use super::super::seq_modules::SequenceRegressor;
use super::super::{Activation, ModuleBuilder};
use std::borrow::Borrow;
use tch::nn;

/// Configuration for a sequence module followed by a regular module
#[derive(Debug, Clone)]
pub struct SequenceRegressorConfig<SC, MC> {
    pub rnn_config: SC,
    pub rnn_hidden_size: usize,
    pub rnn_output_activation: Activation,
    pub post_config: MC,
}

impl<SC, MC> Default for SequenceRegressorConfig<SC, MC>
where
    SC: Default,
    MC: Default,
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

impl<S, SC, M, MC> ModuleBuilder<SequenceRegressor<'static, S, M>>
    for SequenceRegressorConfig<SC, MC>
where
    SC: ModuleBuilder<S>,
    MC: ModuleBuilder<M>,
{
    fn build<'a, P: Borrow<nn::Path<'a>>>(
        &self,
        vs: P,
        input_dim: usize,
        output_dim: usize,
    ) -> SequenceRegressor<'static, S, M> {
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
    use super::super::super::seq_modules::{testing, GruMlp, LstmMlp};
    use super::super::RnnMlpConfig;
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    #[fixture]
    fn gru_mlp_default_module() -> (GruMlp, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = RnnMlpConfig::default();
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

    #[fixture]
    fn lstm_mlp_default_module() -> (LstmMlp, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let config = RnnMlpConfig::default();
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
}
