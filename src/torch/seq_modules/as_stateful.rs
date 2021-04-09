use super::{IterativeModule, SequenceModule, StatefulIterativeModule};
use std::borrow::Borrow;
use tch::Tensor;

/// Wraps an IterativeModule as a StatefulIterativeModule
#[derive(Debug)]
pub struct AsStatefulIterator<M: IterativeModule, T: Borrow<M> = M> {
    pub module: T,
    pub state: M::State,
}

impl<M: IterativeModule, T: Borrow<M>> AsStatefulIterator<M, T> {
    pub fn new(module: T) -> Self {
        let state = module.borrow().initial_state(1);
        Self { module, state }
    }
}

impl<M: IterativeModule> From<M> for AsStatefulIterator<M> {
    fn from(module: M) -> Self {
        Self::new(module)
    }
}

impl<M: IterativeModule, T: Borrow<M>> StatefulIterativeModule for AsStatefulIterator<M, T> {
    fn step(&mut self, input: &Tensor) -> Tensor {
        let (output, new_state) = self.module.borrow().step(&input.unsqueeze(0), &self.state);
        self.state = new_state;
        output.squeeze1(0)
    }
    fn reset(&mut self) {
        self.state = self.module.borrow().initial_state(1);
    }
}

impl<M, T> SequenceModule for AsStatefulIterator<M, T>
where
    M: IterativeModule + SequenceModule,
    T: Borrow<M>,
{
    fn seq_serial(&self, inputs: &Tensor, seq_lengths: &[usize]) -> Tensor {
        self.module.borrow().seq_serial(inputs, seq_lengths)
    }
}

impl<M, T> IterativeModule for AsStatefulIterator<M, T>
where
    M: IterativeModule,
    T: Borrow<M>,
{
    type State = M::State;
    fn initial_state(&self, batch_size: usize) -> Self::State {
        self.module.borrow().initial_state(batch_size)
    }
    fn step(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State) {
        self.module.borrow().step(input, state)
    }
}

#[cfg(test)]
mod as_stateful_module {
    use super::super::{testing, SeqModRnn};
    use super::*;
    use rstest::{fixture, rstest};
    use tch::{nn, Device};

    #[fixture]
    fn linear() -> (AsStatefulIterator<nn::Linear>, usize, usize) {
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
    fn gru() -> (AsStatefulIterator<SeqModRnn<nn::GRU>>, usize, usize) {
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
    fn linear_stateful_step(linear: (AsStatefulIterator<nn::Linear>, usize, usize)) {
        let (mut linear, in_dim, out_dim) = linear;
        testing::check_stateful_step(&mut linear, in_dim, out_dim);
    }
    #[rstest]
    fn linear_seq_serial(linear: (AsStatefulIterator<nn::Linear>, usize, usize)) {
        let (linear, in_dim, out_dim) = linear;
        testing::check_seq_serial(&linear, in_dim, out_dim);
    }

    #[rstest]
    fn linear_step(linear: (AsStatefulIterator<nn::Linear>, usize, usize)) {
        let (linear, in_dim, out_dim) = linear;
        testing::check_step(&linear, in_dim, out_dim);
    }

    #[rstest]
    fn gru_stateful_step(gru: (AsStatefulIterator<SeqModRnn<nn::GRU>>, usize, usize)) {
        let (mut gru, in_dim, out_dim) = gru;
        testing::check_stateful_step(&mut gru, in_dim, out_dim);
    }
    #[rstest]
    fn gru_seq_serial(gru: (AsStatefulIterator<SeqModRnn<nn::GRU>>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_seq_serial(&gru, in_dim, out_dim);
    }

    #[rstest]
    fn gru_step(gru: (AsStatefulIterator<SeqModRnn<nn::GRU>>, usize, usize)) {
        let (gru, in_dim, out_dim) = gru;
        testing::check_step(&gru, in_dim, out_dim);
    }
}
