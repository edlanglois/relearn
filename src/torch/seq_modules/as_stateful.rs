use super::{IterativeModule, StatefulIterativeModule};
use std::borrow::Borrow;
use tch::Tensor;

/// Wraps an IterativeModule as a StatefulIterativeModule
#[derive(Debug)]
pub struct AsStatefulIterator<T: Borrow<M>, M: IterativeModule> {
    pub module: T,
    pub state: M::State,
}

impl<T: Borrow<M>, M: IterativeModule> AsStatefulIterator<T, M> {
    pub fn new(module: T) -> Self {
        let state = module.borrow().initial_state(1);
        Self { module, state }
    }
}

impl<M: IterativeModule> From<M> for AsStatefulIterator<M, M> {
    fn from(module: M) -> Self {
        Self::new(module)
    }
}

impl<T: Borrow<M>, M: IterativeModule> StatefulIterativeModule for AsStatefulIterator<T, M> {
    fn step(&mut self, input: &Tensor) -> Tensor {
        let (output, new_state) = self.module.borrow().step(&input.unsqueeze(0), &self.state);
        self.state = new_state;
        output.squeeze1(0)
    }
    fn reset(&mut self) {
        self.state = self.module.borrow().initial_state(1);
    }
}

#[cfg(test)]
mod as_stateful_module {
    use super::super::{testing, SeqModRnn};
    use super::*;
    use tch::{nn, Device};

    #[test]
    fn linear() {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let mut module: AsStatefulIterator<_, _> = nn::linear(
            &vs.root(),
            in_dim as i64,
            out_dim as i64,
            Default::default(),
        )
        .into();
        testing::check_stateful_step(&mut module, in_dim, out_dim);
    }

    #[test]
    fn gru() {
        let in_dim: usize = 3;
        let out_dim: usize = 2;
        let vs = nn::VarStore::new(Device::Cpu);
        let mut gru: AsStatefulIterator<_, _> = SeqModRnn::from(nn::gru(
            &vs.root(),
            in_dim as i64,
            out_dim as i64,
            Default::default(),
        ))
        .into();
        testing::check_stateful_step(&mut gru, in_dim, out_dim);
    }
}
