use super::super::{Activation, ModuleBuilder};
use super::{IterativeModule, SequenceModule};
use tch::{nn::Func, nn::Module, nn::Path, Tensor};

/// A module stacked on top of a sequence.
pub struct Stacked<'a, T, U> {
    /// The sequence module.
    pub seq: T,
    /// An optional activation function in between
    pub activation: Option<Func<'a>>,
    /// A module applied to the sequence module outputs.
    pub top: U,
}

impl<'a, T, U> Stacked<'a, T, U> {
    pub fn new(seq: T, activation: Option<Func<'a>>, top: U) -> Self {
        Self {
            seq,
            activation,
            top,
        }
    }
}

impl<'a, T, U> SequenceModule for Stacked<'a, T, U>
where
    T: SequenceModule,
    U: Module,
{
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        let mut data = self.seq.seq_serial(inputs, seq_lengths);
        if let Some(ref m) = self.activation {
            data = data.apply(m);
        }
        data.apply(&self.top)
    }

    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor {
        let mut data = self.seq.seq_packed(inputs, batch_sizes);
        if let Some(ref m) = self.activation {
            data = data.apply(m);
        }
        data.apply(&self.top)
    }
}

impl<'a, T, U> IterativeModule for Stacked<'a, T, U>
where
    T: IterativeModule,
    U: Module,
{
    type State = T::State;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        self.seq.initial_state(batch_size)
    }

    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        let (mut data, state) = self.seq.step(input, state);
        if let Some(ref m) = self.activation {
            data = data.apply(m)
        }
        data = data.apply(&self.top);
        (data, state)
    }
}

/// Stacked module configuration
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
    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Stacked<'static, T, U> {
        Stacked::new(
            self.seq_config
                .build_module(&(vs / "rnn"), in_dim, self.seq_output_dim)
                .into(),
            self.seq_output_activation.maybe_module(),
            self.top_config
                .build_module(&(vs / "post"), self.seq_output_dim, out_dim),
        )
    }
}

#[cfg(test)]
mod stacked_module {
    use super::super::{testing, Gru};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    type GruMlp = Stacked<'static, Gru, nn::Linear>;

    /// GRU followed by a relu then a linear layer.
    #[fixture]
    fn gru_relu_linear() -> (GruMlp, usize, usize) {
        let in_dim: usize = 3;
        let hidden_dim: usize = 5;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let path = &vs.root();
        let gru = Gru::new(&(path / "rnn"), in_dim, hidden_dim, true, 0.0);
        let linear = nn::linear(
            &(path / "linear"),
            hidden_dim as i64,
            out_dim as i64,
            Default::default(),
        );
        let sr = Stacked::new(gru, Some(nn::func(Tensor::relu)), linear);
        (sr, in_dim, out_dim)
    }

    #[rstest]
    fn gru_relu_linear_seq_serial(gru_relu_linear: (GruMlp, usize, usize)) {
        let (gru_relu_linear, in_dim, out_dim) = gru_relu_linear;
        testing::check_seq_serial(&gru_relu_linear, in_dim, out_dim);
    }

    #[rstest]
    fn gru_relu_linear_seq_packed(gru_relu_linear: (GruMlp, usize, usize)) {
        let (gru_relu_linear, in_dim, out_dim) = gru_relu_linear;
        testing::check_seq_packed(&gru_relu_linear, in_dim, out_dim);
    }

    #[rstest]
    fn gru_relu_linear_step(gru_relu_linear: (GruMlp, usize, usize)) {
        let (gru_relu_linear, in_dim, out_dim) = gru_relu_linear;
        testing::check_step(&gru_relu_linear, in_dim, out_dim);
    }
}

#[cfg(test)]
mod stacked_module_config {
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
        let module = config.build_module(&vs.root(), in_dim, out_dim);
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
        let module = config.build_module(&vs.root(), in_dim, out_dim);
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
