use super::super::seq_modules::Stacked;
use super::super::{Activation, ModuleBuilder};
use std::borrow::Borrow;
use tch::nn;

/// Configuration of stacked modules.
#[derive(Debug, Clone)]
pub struct StackedConfig<TC, UC> {
    /// Configuration for the initial sequence module layer.
    pub seq_config: TC,
    /// Output dimension of the sequence module layer.
    pub seq_output_dim: usize,
    /// Activation function applied to the sequence module outputs
    pub seq_output_activation: Activation,
    /// Configuration for the module applied on top of the (transformed) sequence outputs.
    pub top_config: UC,
}

impl<TC, UC> Default for StackedConfig<TC, UC>
where
    TC: Default,
    UC: Default,
{
    fn default() -> Self {
        Self {
            seq_config: Default::default(),
            seq_output_dim: 128,
            seq_output_activation: Activation::Relu,
            top_config: Default::default(),
        }
    }
}

impl<T, TC, U, UC> ModuleBuilder<Stacked<'static, T, U>> for StackedConfig<TC, UC>
where
    TC: ModuleBuilder<T>,
    UC: ModuleBuilder<U>,
{
    fn build<'a, P: Borrow<nn::Path<'a>>>(
        &self,
        vs: P,
        input_dim: usize,
        output_dim: usize,
    ) -> Stacked<'static, T, U> {
        let vs = vs.borrow();
        Stacked::new(
            self.seq_config
                .build(vs / "rnn", input_dim, self.seq_output_dim)
                .into(),
            self.seq_output_activation.maybe_module(),
            self.top_config
                .build(vs / "post", self.seq_output_dim, output_dim),
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
