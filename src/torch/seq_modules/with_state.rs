use super::super::BuildModule;
use super::{IterativeModule, SequenceModule, StatefulIterativeModule};
use crate::torch::backends::CudnnSupport;
use std::borrow::Borrow;
use tch::{nn::Path, Tensor};

/// Configuration for [`WithState`].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct WithStateConfig<T> {
    pub module_config: T,
}

impl<T> From<T> for WithStateConfig<T> {
    fn from(module_config: T) -> Self {
        Self { module_config }
    }
}

impl<T> BuildModule for WithStateConfig<T>
where
    T: BuildModule,
    T::Module: IterativeModule,
{
    type Module = WithState<T::Module>;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        self.module_config.build_module(vs, in_dim, out_dim).into()
    }
}

/// [`IterativeModule`] wrapper that also stores the state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WithState<T: IterativeModule> {
    pub module: T,
    pub state: T::State,
}

impl<T: IterativeModule> WithState<T> {
    pub fn new(module: T) -> Self {
        let state = module.initial_state(1); // batch size of 1
        Self { module, state }
    }
}

impl<T: IterativeModule> From<T> for WithState<T> {
    fn from(module: T) -> Self {
        Self::new(module)
    }
}

impl<T: IterativeModule> StatefulIterativeModule for WithState<T> {
    fn step(&mut self, input: &Tensor) -> Tensor {
        let (output, new_state) = self.module.step(&input.unsqueeze(0), &self.state);
        self.state = new_state;
        output.squeeze_dim(0)
    }
    fn reset(&mut self) {
        self.state = self.module.initial_state(1);
    }
}

impl<T> SequenceModule for WithState<T>
where
    T: IterativeModule + SequenceModule,
{
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        self.module.borrow().seq_serial(inputs, seq_lengths)
    }

    fn seq_packed(&self, inputs: &Tensor, batch_sizes: &Tensor) -> Tensor {
        self.module.borrow().seq_packed(inputs, batch_sizes)
    }
}

// TODO: Consider deleting this.
// It can lead the compiler to try an infinite regression of WithState<WithState<...>>
//
// As an alternative, could implement Deref<T> for WithState<T>.
impl<T> IterativeModule for WithState<T>
where
    T: IterativeModule,
{
    type State = T::State;
    fn initial_state(&self, batch_size: usize) -> Self::State {
        self.module.initial_state(batch_size)
    }
    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        self.module.step(input, state)
    }
}

impl<T> CudnnSupport for WithState<T>
where
    T: IterativeModule + CudnnSupport,
{
    fn has_cudnn_second_derivatives(&self) -> bool {
        self.module.has_cudnn_second_derivatives()
    }
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod with_state {
    use super::super::{testing, Gru, GruConfig, MlpConfig};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, nn::LinearConfig, Device};

    #[fixture]
    fn linear() -> (WithState<nn::Linear>, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let linear = nn::linear(
            &vs.root(),
            in_dim as i64,
            out_dim as i64,
            LinearConfig::default(),
        );
        (linear.into(), in_dim, out_dim)
    }

    #[fixture]
    fn gru() -> (WithState<Gru>, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let gru = Gru::new(&vs.root(), in_dim, out_dim, true, 0.0);
        (gru.into(), in_dim, out_dim)
    }

    #[rstest]
    fn linear_stateful_step(linear: (WithState<nn::Linear>, usize, usize)) {
        let (mut linear, in_dim, out_dim) = linear;
        testing::check_stateful_step(&mut linear, in_dim, out_dim);
    }
    #[rstest]
    fn linear_seq_serial(linear: (WithState<nn::Linear>, usize, usize)) {
        let (linear, in_dim, out_dim) = linear;
        testing::check_seq_serial(&linear, in_dim, out_dim);
    }
    #[rstest]
    fn linear_seq_packed(linear: (WithState<nn::Linear>, usize, usize)) {
        let (linear, in_dim, out_dim) = linear;
        testing::check_seq_packed(&linear, in_dim, out_dim);
    }

    #[rstest]
    fn linear_step(linear: (WithState<nn::Linear>, usize, usize)) {
        let (linear, in_dim, out_dim) = linear;
        testing::check_step(&linear, in_dim, out_dim);
    }

    #[test]
    fn linear_module_builder() {
        let config = WithStateConfig::from(MlpConfig::default());
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = config.build_module(&vs.root(), 1, 1);
    }

    #[rstest]
    fn gru_stateful_step(gru: (WithState<Gru>, usize, usize)) {
        let (mut gru, in_dim, out_dim) = gru;
        testing::check_stateful_step(&mut gru, in_dim, out_dim);
    }
    #[rstest]
    fn gru_seq_serial(gru: (WithState<Gru>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_serial(&gru, in_dim, out_dim);
    }
    #[rstest]
    fn gru_seq_packed(gru: (WithState<Gru>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_packed(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_step(gru: (WithState<Gru>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_step(&gru, in_dim, out_dim);
    }

    #[test]
    fn gru_module_builder() {
        let config = WithStateConfig::from(GruConfig::default());
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = config.build_module(&vs.root(), 1, 1);
    }
}
