//! Modules applied one after another in sequence
use super::{Activation, BuildModule, FeedForwardModule, IterativeModule, Module, SequenceModule};
use tch::{nn::Path, Tensor};

/// Configuration for a [`Chained`] module.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ChainedConfig<T, U> {
    pub first_config: T,
    pub second_config: U,
    pub hidden_dim: usize,
    pub activation: Activation,
}

impl<T, U> Default for ChainedConfig<T, U>
where
    T: Default,
    U: Default,
{
    fn default() -> Self {
        Self {
            first_config: T::default(),
            second_config: U::default(),
            hidden_dim: 128,
            activation: Activation::default(),
        }
    }
}

impl<T, U> BuildModule for ChainedConfig<T, U>
where
    T: BuildModule,
    U: BuildModule,
{
    type Module = Chained<T::Module, U::Module>;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        Chained {
            first: self
                .first_config
                .build_module(&(vs / "0"), in_dim, self.hidden_dim),
            second: self
                .second_config
                .build_module(&(vs / "1"), self.hidden_dim, out_dim),
            activation: self.activation,
        }
    }
}

/// One module applied to the output of another with an optional activation function in between.
#[derive(Debug, Default, Copy, Clone)]
pub struct Chained<T, U> {
    pub first: T,
    pub second: U,
    pub activation: Activation,
}

impl<T, U> Chained<T, U> {
    pub const fn new(first: T, second: U, activation: Activation) -> Self {
        Self {
            first,
            second,
            activation,
        }
    }
}

impl<T, U> Module for Chained<T, U>
where
    T: Module,
    U: Module,
{
    fn has_cudnn_second_derivatives(&self) -> bool {
        self.first.has_cudnn_second_derivatives() && self.second.has_cudnn_second_derivatives()
    }
}

impl<T, U> FeedForwardModule for Chained<T, U>
where
    T: FeedForwardModule,
    U: FeedForwardModule,
{
    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = self.first.forward(input);
        let hidden = self.activation.apply(hidden);
        self.second.forward(&hidden)
    }
}

impl<T, U> SequenceModule for Chained<T, U>
where
    T: SequenceModule,
    U: SequenceModule,
{
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        let hidden = self.first.seq_serial(inputs, seq_lengths);
        let hidden = self.activation.apply(hidden);
        self.second.seq_serial(&hidden, seq_lengths)
    }
    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor {
        let hidden = self.first.seq_packed(inputs, batch_sizes);
        let hidden = self.activation.apply(hidden);
        self.second.seq_packed(&hidden, batch_sizes)
    }
}

impl<T, U> IterativeModule for Chained<T, U>
where
    T: IterativeModule,
    U: IterativeModule,
{
    type State = (T::State, U::State);

    fn initial_state(&self) -> Self::State {
        (self.first.initial_state(), self.second.initial_state())
    }

    fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor {
        let hidden = self.first.step(&mut state.0, input);
        let hidden = self.activation.apply(hidden);
        self.second.step(&mut state.1, &hidden)
    }
}

/* TODO: Remove?
/// Configuration for a stack of modules each applied to the output of the previous.
pub struct StackedConfig<MC> {
    // First layer configuration
    first_config: MC,
    // (hidden_size, module_config) for each following layer
    rest_configs: Vec<(usize, MC)>,
}

impl<MC> StackedConfig<MC> {
    /// Create a new stacked configuration containing just the first layer
    pub fn new<T: Into<MC>>(config: T) -> Self {
        Self {
            first_config: config.into(),
            rest_configs: Vec::new(),
        }
    }

    /// Push a module configuration on second of the stack.
    pub fn push<T: Into<MC>>(&mut self, hidden_dim: usize, config: T) {
        self.rest_configs.push((hidden_dim, config.into()))
    }
}

impl<MC: BuildModule> BuildModule for StackedConfig<MC> {
    type Module = Vec<MC::Module>;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        iter::once((in_dim, &self.first_config))
            .chain(self.rest_configs.iter().map(|(in_, config)| (*in_, config)))
            .zip(
                self.rest_configs
                    .iter()
                    .map(|(out, _)| *out)
                    .chain(iter::once(out_dim)),
            )
            .enumerate()
            .map(|(i, ((in_, config), out))| {
                config.build_module(&(vs / format!("layer_{}", i)), in_, out)
            })
            .collect()
    }
}
*/

impl<M: Module> Module for [M] {
    fn has_cudnn_second_derivatives(&self) -> bool {
        self.iter().all(M::has_cudnn_second_derivatives)
    }
}

impl<M: Module, const N: usize> Module for [M; N] {
    fn has_cudnn_second_derivatives(&self) -> bool {
        self.iter().all(M::has_cudnn_second_derivatives)
    }
}

impl<M: FeedForwardModule> FeedForwardModule for [M] {
    fn forward(&self, input: &Tensor) -> Tensor {
        fold_or_clone(self, input, |tensor, module| module.forward(tensor))
    }
}

impl<M: FeedForwardModule, const N: usize> FeedForwardModule for [M; N] {
    fn forward(&self, input: &Tensor) -> Tensor {
        fold_or_clone(self, input, |tensor, module| module.forward(tensor))
    }
}

impl<M: SequenceModule> SequenceModule for [M] {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        fold_or_clone(self, inputs, |tensor, module| {
            module.seq_serial(tensor, seq_lengths)
        })
    }

    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor {
        fold_or_clone(self, inputs, |tensor, module| {
            module.seq_packed(tensor, batch_sizes)
        })
    }
}

impl<M: SequenceModule, const N: usize> SequenceModule for [M; N] {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        fold_or_clone(self, inputs, |tensor, module| {
            module.seq_serial(tensor, seq_lengths)
        })
    }

    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor {
        fold_or_clone(self, inputs, |tensor, module| {
            module.seq_packed(tensor, batch_sizes)
        })
    }
}

impl<M: IterativeModule> IterativeModule for [M] {
    type State = Vec<M::State>;

    fn initial_state(&self) -> Self::State {
        self.iter().map(M::initial_state).collect()
    }

    fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor {
        assert_eq!(self.len(), state.len(), "mismatched state lenght");
        fold_or_clone(
            self.iter().zip(state.iter_mut()),
            input,
            |tensor, (module, module_state)| module.step(module_state, tensor),
        )
    }
}

impl<M: IterativeModule, const N: usize> IterativeModule for [M; N] {
    type State = [M::State; N];

    fn initial_state(&self) -> Self::State {
        array_init::from_iter(self.iter().map(M::initial_state)).unwrap()
    }

    fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor {
        fold_or_clone(
            self.iter().zip(state.iter_mut()),
            input,
            |tensor, (module, module_state)| module.step(module_state, tensor),
        )
    }
}

/// Either fold an iterator over an input or clone the input Tensor if the iterator is empty
fn fold_or_clone<I, F>(modules: I, input: &Tensor, mut f: F) -> Tensor
where
    I: IntoIterator,
    F: FnMut(&Tensor, I::Item) -> Tensor,
{
    let mut iter = modules.into_iter();
    let tensor = match iter.next() {
        Some(module) => f(input, module),
        None => return input.shallow_clone(),
    };
    iter.fold(tensor, |t, m| f(&t, m))
}

#[cfg(test)]
mod tests {
    use super::super::{testing, AsSeq, Gru, GruConfig, Mlp, MlpConfig};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn::VarStore, Device, Kind};

    fn chained_mlp_config() -> ChainedConfig<MlpConfig, MlpConfig> {
        let mlp_config = MlpConfig {
            hidden_sizes: vec![16],
            ..MlpConfig::default()
        };
        ChainedConfig {
            first_config: mlp_config.clone(),
            second_config: mlp_config,
            hidden_dim: 8,
            ..ChainedConfig::default()
        }
    }

    fn chained_gru_mlp_config() -> ChainedConfig<GruConfig, AsSeq<MlpConfig>> {
        ChainedConfig {
            first_config: GruConfig::default(),
            second_config: AsSeq::new(MlpConfig {
                hidden_sizes: vec![16],
                ..MlpConfig::default()
            }),
            hidden_dim: 8,
            ..ChainedConfig::default()
        }
    }

    #[fixture]
    fn chained_mlp() -> (Chained<Mlp, Mlp>, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let vs = VarStore::new(Device::Cpu);
        let mlp = chained_mlp_config().build_module(&vs.root(), in_dim, out_dim);
        (mlp, in_dim, out_dim)
    }

    #[fixture]
    fn gru_mlp() -> (Chained<Gru, AsSeq<Mlp>>, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let vs = VarStore::new(Device::Cpu);
        let mlp = chained_gru_mlp_config().build_module(&vs.root(), in_dim, out_dim);
        (mlp, in_dim, out_dim)
    }

    #[rstest]
    fn chained_mlp_forward(chained_mlp: (Chained<Mlp, Mlp>, usize, usize)) {
        let (chained_mlp, in_dim, out_dim) = chained_mlp;
        testing::check_forward(&chained_mlp, in_dim, out_dim, &[4], Kind::Float);
    }

    #[rstest]
    fn gru_mlp_seq_serial(gru_mlp: (Chained<Gru, AsSeq<Mlp>>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_seq_serial(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_seq_packed(gru_mlp: (Chained<Gru, AsSeq<Mlp>>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_seq_packed(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_step(gru_mlp: (Chained<Gru, AsSeq<Mlp>>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_step(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_seq_packed_matches_iter_steps(gru_mlp: (Chained<Gru, AsSeq<Mlp>>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_seq_packed_matches_iter_steps(&gru_mlp, in_dim, out_dim);
    }

    #[test]
    fn chained_mlp_forward_gradient_descent() {
        testing::check_config_forward_gradient_descent(&chained_mlp_config());
    }

    #[test]
    fn gru_mlp_seq_packed_gradient_descent() {
        testing::check_config_seq_packed_gradient_descent(&chained_gru_mlp_config());
    }
}