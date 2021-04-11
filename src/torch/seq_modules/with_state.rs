use super::{IterativeModule, SequenceModule, StatefulIterativeModule};
use std::borrow::Borrow;
use tch::Tensor;

/// IterativeModule wrapper that also stores the state.
#[derive(Debug)]
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
        output.squeeze1(0)
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
}

// Note: Consider deleting this.
// It is currently unused and can lead the compiler to try
// an infinite regression of WithState<WithState<...>>
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

#[cfg(test)]
mod as_stateful_module {
    use super::super::{testing, SeqModRnn};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    #[fixture]
    fn linear() -> (WithState<nn::Linear>, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let linear = nn::linear(
            &vs.root(),
            in_dim as i64,
            out_dim as i64,
            Default::default(),
        );
        (linear.into(), in_dim, out_dim)
    }

    #[fixture]
    fn gru() -> (WithState<SeqModRnn<nn::GRU>>, usize, usize) {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let gru = SeqModRnn::from(nn::gru(
            &vs.root(),
            in_dim as i64,
            out_dim as i64,
            Default::default(),
        ));
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
    fn linear_step(linear: (WithState<nn::Linear>, usize, usize)) {
        let (linear, in_dim, out_dim) = linear;
        testing::check_step(&linear, in_dim, out_dim);
    }

    #[rstest]
    fn gru_stateful_step(gru: (WithState<SeqModRnn<nn::GRU>>, usize, usize)) {
        let (mut gru, in_dim, out_dim) = gru;
        testing::check_stateful_step(&mut gru, in_dim, out_dim);
    }
    #[rstest]
    fn gru_seq_serial(gru: (WithState<SeqModRnn<nn::GRU>>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_serial(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_step(gru: (WithState<SeqModRnn<nn::GRU>>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_step(&gru, in_dim, out_dim);
    }
}
