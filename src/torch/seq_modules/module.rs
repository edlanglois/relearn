//! Implement sequence module traits for feed-forward modules.
use super::{IterativeModule, SequenceModule, StatefulIterativeModule};
use tch::{nn::Module, Tensor};

impl<M: Module> SequenceModule for M {
    fn seq_serial(&self, inputs: &Tensor, _seq_lengths: &[usize]) -> Tensor {
        inputs.apply(self)
    }

    fn seq_packed(&self, inputs: &Tensor, _batch_sizes: &Tensor) -> Tensor {
        inputs.apply(self)
    }
}

impl<M: Module> IterativeModule for M {
    type State = ();

    fn initial_state(&self, _batch_size: usize) -> Self::State {}

    fn step(&self, input: &Tensor, _state: &Self::State) -> (Tensor, Self::State) {
        (input.apply(self), ())
    }
}

impl<M: Module> StatefulIterativeModule for M {
    fn step(&mut self, input: &Tensor) -> Tensor {
        input.apply(self)
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod sequence_module {
    use super::*;
    use tch::kind::Kind;
    use tch::nn;

    #[test]
    fn seq_serial_batched() {
        let m = nn::func(|x| x * 2);
        let batch_size = 2;
        let seq_len = 3;
        let feature_dim = 1;
        let input = Tensor::of_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).view((
            batch_size,
            seq_len,
            feature_dim,
        ));
        let seq_lengths = [2, 1];
        let output = m.seq_serial(&input, &seq_lengths);

        let expected = Tensor::of_slice(&[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]).view((
            batch_size,
            seq_len,
            feature_dim,
        ));
        assert_eq!(output, expected);
    }

    #[test]
    fn seq_serial_feature_dim_changed() {
        let m = nn::func(|x| x.sum1(&[-1], true, Kind::Float));
        let batch_size = 1;
        let seq_len = 3;
        let in_features = 2;
        let out_features = 1;
        let input = Tensor::of_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).view((
            batch_size,
            seq_len,
            in_features,
        ));
        let seq_lengths = [2, 1];
        let output = m.seq_serial(&input, &seq_lengths);

        let expected = Tensor::of_slice(&[1.0, 5.0, 9.0]).view((batch_size, seq_len, out_features));
        assert_eq!(output, expected);
    }

    #[test]
    fn seq_packed_batched() {
        let m = nn::func(|x| x * 2);
        let feature_dim = 1;
        let input = Tensor::of_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).view((-1, feature_dim));
        let batch_sizes = Tensor::of_slice(&[3_i64, 2, 1]);
        let output = m.seq_packed(&input, &batch_sizes);

        let expected = Tensor::of_slice(&[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]).view((-1, feature_dim));
        assert_eq!(output, expected);
    }
}

#[cfg(test)]
#[allow(clippy::let_unit_value)] // state is () but this is conceptually unimportant
mod iterative_module {
    use super::*;
    use tch::kind::Kind;
    use tch::nn;

    #[test]
    fn batched() {
        let m = nn::func(|x| x * 2);
        let batch_size: usize = 2;
        let feature_dim = 3;
        let input = Tensor::of_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .view((batch_size as i64, feature_dim));

        let state = m.initial_state(batch_size);
        let (output, _) = m.step(&input, &state);

        let expected = Tensor::of_slice(&[0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
            .view((batch_size as i64, feature_dim));
        assert_eq!(output, expected);
    }

    #[test]
    fn feature_dim_changed() {
        let m = nn::func(|x| x.sum1(&[-1], true, Kind::Float));
        let batch_size: usize = 1;
        let input = Tensor::of_slice(&[0.0, 1.0, 2.0]).view((batch_size as i64, 3));

        let state = m.initial_state(batch_size);
        let (output, _) = m.step(&input, &state);

        let expected = Tensor::of_slice(&[3.0]).view((batch_size as i64, 1));
        assert_eq!(output, expected);
    }
}
