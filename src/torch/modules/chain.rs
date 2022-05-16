//! Modules applied one after another in sequence
use super::{
    Activation, BuildModule, FeedForwardModule, IterativeModule, Module, ModuleExtras,
    SequenceModule,
};
use crate::torch::packed::PackedTensor;
use serde::{Deserialize, Serialize};
use std::iter;
use tch::{Device, Tensor};

/// Configuration for a [`Chain`] module.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChainConfig<A, B> {
    pub first_config: A,
    pub second_config: B,
    pub hidden_dim: usize,
    pub activation: Activation,
}

impl<A, B> Default for ChainConfig<A, B>
where
    A: Default,
    B: Default,
{
    fn default() -> Self {
        Self {
            first_config: A::default(),
            second_config: B::default(),
            hidden_dim: 128,
            activation: Activation::default(),
        }
    }
}

impl<A, B> BuildModule for ChainConfig<A, B>
where
    A: BuildModule,
    A::Module: for<'a> ModuleExtras<'a>,
    B: BuildModule,
    B::Module: for<'a> ModuleExtras<'a>,
{
    type Module = Chain<A::Module, B::Module>;

    fn build_module(&self, in_dim: usize, out_dim: usize, device: Device) -> Self::Module {
        Chain {
            first: self
                .first_config
                .build_module(in_dim, self.hidden_dim, device),
            second: self
                .second_config
                .build_module(self.hidden_dim, out_dim, device),
            activation: self.activation,
        }
    }
}

/// One module applied to the output of another with an optional activation function in between.
#[derive(Debug, Default, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chain<A, B> {
    pub first: A,
    pub second: B,
    pub activation: Activation,
}

impl<A, B> Chain<A, B> {
    pub const fn new(first: A, second: B, activation: Activation) -> Self {
        Self {
            first,
            second,
            activation,
        }
    }
}

impl<A, B> Module for Chain<A, B>
where
    A: Module + for<'a> ModuleExtras<'a>,
    B: Module + for<'a> ModuleExtras<'a>,
{
    fn shallow_clone(&self) -> Self
    where
        Self: Sized,
    {
        Self {
            first: self.first.shallow_clone(),
            second: self.second.shallow_clone(),
            ..*self
        }
    }

    fn clone_to_device(&self, device: Device) -> Self
    where
        Self: Sized,
    {
        Self {
            first: self.first.clone_to_device(device),
            second: self.second.clone_to_device(device),
            ..*self
        }
    }

    #[inline]
    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(ModuleExtras::variables(self))
    }

    #[inline]
    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(ModuleExtras::trainable_variables(self))
    }

    fn has_cudnn_second_derivatives(&self) -> bool {
        self.first.has_cudnn_second_derivatives() && self.second.has_cudnn_second_derivatives()
    }
}

impl<'a, A, B> ModuleExtras<'a> for Chain<A, B>
where
    A: ModuleExtras<'a>,
    B: ModuleExtras<'a>,
{
    type Variables = iter::Chain<A::Variables, B::Variables>;
    type TrainableVariables = iter::Chain<A::TrainableVariables, B::TrainableVariables>;

    fn variables(&'a self) -> Self::Variables {
        self.first.variables().chain(self.second.variables())
    }
    fn trainable_variables(&'a self) -> Self::TrainableVariables {
        self.first
            .trainable_variables()
            .chain(self.second.trainable_variables())
    }
}

impl<A, B> FeedForwardModule for Chain<A, B>
where
    A: FeedForwardModule + for<'a> ModuleExtras<'a>,
    B: FeedForwardModule + for<'a> ModuleExtras<'a>,
{
    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = self.first.forward(input);
        let hidden = self.activation.forward_owned(hidden);
        self.second.forward(&hidden)
    }
}

impl<A, B> SequenceModule for Chain<A, B>
where
    A: SequenceModule + for<'a> ModuleExtras<'a>,
    B: SequenceModule + for<'a> ModuleExtras<'a>,
{
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        let hidden = self.first.seq_serial(inputs, seq_lengths);
        let hidden = self.activation.forward_owned(hidden);
        self.second.seq_serial(&hidden, seq_lengths)
    }
    fn seq_packed(&self, inputs: &PackedTensor) -> PackedTensor {
        let hidden = self.first.seq_packed(inputs);
        let hidden = hidden.batch_map(|tensor| self.activation.forward_owned(tensor));
        self.second.seq_packed(&hidden)
    }
}

impl<A, B> IterativeModule for Chain<A, B>
where
    A: IterativeModule + for<'a> ModuleExtras<'a>,
    B: IterativeModule + for<'a> ModuleExtras<'a>,
{
    type State = (A::State, B::State);

    fn initial_state(&self) -> Self::State {
        (self.first.initial_state(), self.second.initial_state())
    }

    fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor {
        let hidden = self.first.step(&mut state.0, input);
        let hidden = self.activation.forward_owned(hidden);
        self.second.step(&mut state.1, &hidden)
    }
}

impl<M: Module> Module for [M] {
    fn shallow_clone(&self) -> Self
    where
        Self: Sized,
    {
        // TODO: Why is this implementation expected? Is [M] not unsized?
        unimplemented!()
    }

    fn clone_to_device(&self, _: Device) -> Self
    where
        Self: Sized,
    {
        // TODO: Why is this implementation expected? Is [M] not unsized?
        unimplemented!()
    }

    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(self.iter().flat_map(Module::variables))
    }

    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(self.iter().flat_map(Module::trainable_variables))
    }

    fn has_cudnn_second_derivatives(&self) -> bool {
        self.iter().all(M::has_cudnn_second_derivatives)
    }
}

impl<M: Module, const N: usize> Module for [M; N] {
    fn shallow_clone(&self) -> Self
    where
        Self: Sized,
    {
        array_init::array_init(|i| self[i].shallow_clone())
    }

    fn clone_to_device(&self, device: Device) -> Self
    where
        Self: Sized,
    {
        array_init::array_init(|i| self[i].clone_to_device(device))
    }

    fn variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(self.iter().flat_map(Module::variables))
    }

    fn trainable_variables(&self) -> Box<dyn Iterator<Item = &Tensor> + '_> {
        Box::new(self.iter().flat_map(Module::trainable_variables))
    }

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

    fn seq_packed(&self, inputs: &PackedTensor) -> PackedTensor {
        fold_or_clone(self, inputs, |packed_tensor, module| {
            module.seq_packed(packed_tensor)
        })
    }
}

impl<M: SequenceModule, const N: usize> SequenceModule for [M; N] {
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        fold_or_clone(self, inputs, |tensor, module| {
            module.seq_serial(tensor, seq_lengths)
        })
    }

    fn seq_packed(&self, inputs: &PackedTensor) -> PackedTensor {
        fold_or_clone(self, inputs, |packed_tensor, module| {
            module.seq_packed(packed_tensor)
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
        array_init::from_iter(<[M]>::iter(self).map(M::initial_state)).unwrap()
    }

    fn step(&self, state: &mut Self::State, input: &Tensor) -> Tensor {
        fold_or_clone(
            <[M]>::iter(self).zip(state.iter_mut()),
            input,
            |tensor, (module, module_state)| module.step(module_state, tensor),
        )
    }
}

/// Either fold an iterator over an input or clone the input Tensor if the iterator is empty
fn fold_or_clone<I, T, F>(modules: I, input: &T, mut f: F) -> T
where
    I: IntoIterator,
    T: Clone_,
    F: FnMut(&T, I::Item) -> T,
{
    let mut iter = modules.into_iter();
    let tensor = match iter.next() {
        Some(module) => f(input, module),
        None => return input.clone_(),
    };
    iter.fold(tensor, |t, m| f(&t, m))
}

/// Like [`Clone`] but includes [`Tensor::shallow_clone`].
trait Clone_ {
    fn clone_(&self) -> Self;
}

impl Clone_ for Tensor {
    fn clone_(&self) -> Self {
        self.shallow_clone()
    }
}
impl Clone_ for PackedTensor {
    fn clone_(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
// Confusion with rstest hack when passing the _runner arg
#[allow(
    clippy::needless_pass_by_value,
    clippy::used_underscore_binding,
    clippy::no_effect_underscore_binding
)]
mod tests {
    use super::super::testing::{
        self, RunForward, RunIterStep, RunModule, RunSeqPacked, RunSeqSerial,
    };
    use super::super::{Gru, GruConfig, Mlp, MlpConfig};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{Device, Kind};

    fn chained_mlp_config() -> ChainConfig<MlpConfig, MlpConfig> {
        let mlp_config = MlpConfig {
            hidden_sizes: vec![16],
            ..MlpConfig::default()
        };
        ChainConfig {
            first_config: mlp_config.clone(),
            second_config: mlp_config,
            hidden_dim: 8,
            ..ChainConfig::default()
        }
    }

    fn chained_gru_mlp_config() -> ChainConfig<GruConfig, MlpConfig> {
        ChainConfig {
            first_config: GruConfig::default(),
            second_config: MlpConfig {
                hidden_sizes: vec![16],
                ..MlpConfig::default()
            },
            hidden_dim: 8,
            ..ChainConfig::default()
        }
    }

    #[fixture]
    fn chained_mlp() -> (Chain<Mlp, Mlp>, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let mlp = chained_mlp_config().build_module(in_dim, out_dim, Device::Cpu);
        (mlp, in_dim, out_dim)
    }

    #[fixture]
    fn gru_mlp() -> (Chain<Gru, Mlp>, usize, usize) {
        let in_dim = 3;
        let out_dim = 2;
        let mlp = chained_gru_mlp_config().build_module(in_dim, out_dim, Device::Cpu);
        (mlp, in_dim, out_dim)
    }

    #[rstest]
    fn chained_mlp_forward(chained_mlp: (Chain<Mlp, Mlp>, usize, usize)) {
        let (chained_mlp, in_dim, out_dim) = chained_mlp;
        testing::check_forward(&chained_mlp, in_dim, out_dim, &[4], Kind::Float);
    }

    #[rstest]
    fn gru_mlp_seq_serial(gru_mlp: (Chain<Gru, Mlp>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_seq_serial(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_seq_packed(gru_mlp: (Chain<Gru, Mlp>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_seq_packed(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_step(gru_mlp: (Chain<Gru, Mlp>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_step(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    fn gru_mlp_seq_packed_matches_iter_steps(gru_mlp: (Chain<Gru, Mlp>, usize, usize)) {
        let (gru_mlp, in_dim, out_dim) = gru_mlp;
        testing::check_seq_packed_matches_iter_steps(&gru_mlp, in_dim, out_dim);
    }

    #[rstest]
    #[case::forward(RunForward)]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn chained_mlp_gradient_descent<R: RunModule<Chain<Mlp, Mlp>>>(#[case] _runner: R) {
        testing::check_config_gradient_descent::<R, _>(&chained_mlp_config());
    }

    #[rstest]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn gru_mlp_gradient_descent<R: RunModule<Chain<Gru, Mlp>>>(#[case] _runner: R) {
        testing::check_config_gradient_descent::<R, _>(&chained_gru_mlp_config());
    }

    #[rstest]
    #[case::forward(RunForward)]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn chained_mlp_clone_to_new_device<R: RunModule<Chain<Mlp, Mlp>>>(#[case] _runner: R) {
        testing::check_config_clone_to_new_device::<R, _>(&chained_mlp_config());
    }

    #[rstest]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn gru_mlp_clone_to_new_device<R: RunModule<Chain<Gru, Mlp>>>(#[case] _runner: R) {
        testing::check_config_clone_to_new_device::<R, _>(&chained_gru_mlp_config());
    }

    #[test]
    fn chained_mlp_clone_to_same_device() {
        testing::check_config_clone_to_same_device::<RunForward, _>(&chained_mlp_config());
    }

    #[test]
    fn gru_mlp_clone_to_same_device() {
        testing::check_config_clone_to_same_device::<RunSeqPacked, _>(&chained_gru_mlp_config());
    }

    #[rstest]
    #[case::forward(RunForward)]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn chained_mlp_ser_de_matches<R: RunModule<Chain<Mlp, Mlp>>>(
        #[case] _runner: R,
        chained_mlp: (Chain<Mlp, Mlp>, usize, usize),
    ) {
        let (module, in_dim, _) = chained_mlp;
        testing::check_ser_de_matches::<R, _>(&module, in_dim);
    }

    #[rstest]
    #[case::seq_serial(RunSeqSerial)]
    #[case::seq_packed(RunSeqPacked)]
    #[case::iter_step(RunIterStep)]
    fn gru_mlp_ser_de_matches<R: RunModule<Chain<Gru, Mlp>>>(
        #[case] _runner: R,
        gru_mlp: (Chain<Gru, Mlp>, usize, usize),
    ) {
        let (module, in_dim, _) = gru_mlp;
        testing::check_ser_de_matches::<R, _>(&module, in_dim);
    }

    #[rstest]
    fn variables_count(gru_mlp: (Chain<Gru, Mlp>, usize, usize)) {
        let (gru_mlp, _, _) = gru_mlp;
        assert_eq!(Module::variables(&gru_mlp).count(), 8);
    }

    #[rstest]
    fn trainable_variables_count(gru_mlp: (Chain<Gru, Mlp>, usize, usize)) {
        let (gru_mlp, _, _) = gru_mlp;
        assert_eq!(Module::trainable_variables(&gru_mlp).count(), 8);
    }
}
