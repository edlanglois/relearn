//! Sequence / Iterative module view of a feed-forward module.
use super::super::{BuildModule, FeedForwardModule, IterativeModule, Module, SequenceModule};
use tch::{nn::Path, Tensor};

/// Wraps a [`FeedForwardModule`] as a [`SequenceModule`] or [`IterativeModule`] by batching.
///
/// Treats the sequence dimension as a batch dimension, so each sequence step is transformed
/// identically and independently.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AsSeq<M> {
    pub inner: M,
}

impl<M> AsSeq<M> {
    pub const fn new(module: M) -> Self {
        Self { inner: module }
    }
}

impl<M> From<M> for AsSeq<M> {
    fn from(inner: M) -> Self {
        Self { inner }
    }
}

impl<M: Module> Module for AsSeq<M> {
    fn has_cudnn_second_derivatives(&self) -> bool {
        self.inner.has_cudnn_second_derivatives()
    }
}

impl<MC: BuildModule> BuildModule for AsSeq<MC> {
    type Module = AsSeq<MC::Module>;

    fn build_module(&self, vs: &Path, in_dim: usize, out_dim: usize) -> Self::Module {
        AsSeq {
            inner: self.inner.build_module(vs, in_dim, out_dim),
        }
    }
}

impl<M: FeedForwardModule> FeedForwardModule for AsSeq<M> {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.inner.forward(input)
    }
}

impl<M: FeedForwardModule> SequenceModule for AsSeq<M> {
    fn seq_serial(&self, inputs: &Tensor, _seq_lengths: &[usize]) -> Tensor {
        self.inner.forward(inputs)
    }

    fn seq_packed(&self, inputs: &Tensor, _batch_sizes: &Tensor) -> Tensor {
        self.inner.forward(inputs)
    }
}

impl<M: FeedForwardModule> IterativeModule for AsSeq<M> {
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn step(&self, _: &mut Self::State, input: &Tensor) -> Tensor {
        self.inner.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{testing, MlpConfig};
    use super::*;

    #[test]
    fn as_seq_mlp_forward_gradient_descent() {
        let mlp_config = MlpConfig {
            hidden_sizes: vec![16],
            ..MlpConfig::default()
        };
        let config = AsSeq::new(mlp_config);
        testing::check_config_forward_gradient_descent(&config);
    }

    #[test]
    fn as_seq_mlp_seq_packed_gradient_descent() {
        let mlp_config = MlpConfig {
            hidden_sizes: vec![16],
            ..MlpConfig::default()
        };
        let config = AsSeq::new(mlp_config);
        testing::check_config_seq_packed_gradient_descent(&config);
    }
}
