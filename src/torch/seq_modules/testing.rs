//! Sequence modules test utilities.
use super::{IterativeModule, SequenceModule};
use std::iter;
use tch::{kind::Kind, nn::Module, Device, IndexOp, Tensor};

/// Basic structural check of [tch::nn::Module::forward] via Tensor::apply
pub fn check_forward<M: Module>(
    module: &M,
    in_dim: usize,
    out_dim: usize,
    batch_shape: &[usize],
    kind: Kind,
) {
    let input_shape: Vec<_> = batch_shape
        .into_iter()
        .chain(iter::once(&in_dim))
        .map(|&d| d as i64)
        .collect();
    let input = Tensor::ones(&input_shape, (kind, Device::Cpu));
    let output = input.apply(module);
    let mut output_shape = input_shape;
    *output_shape.last_mut().unwrap() = out_dim as i64;
    assert_eq!(output.size(), output_shape);
}

/// Basic check of SequenceModule::seq_serial
///
/// * Checks that the output size is correct.
/// * Checks that identical inner sequences produce identical output.
pub fn check_seq_serial<M: SequenceModule>(module: &M, in_dim: usize, out_dim: usize) {
    let batch_size: usize = 4;

    // Step indices by sequence: 0 | 1 2 3 | 4 5
    let seq_lengths: [usize; 3] = [1, 3, 2];
    let total_num_steps: usize = seq_lengths.iter().sum();

    let inputs = Tensor::ones(
        &[batch_size as i64, total_num_steps as i64, in_dim as i64],
        (Kind::Float, Device::Cpu),
    );

    let output = module.seq_serial(&inputs, &seq_lengths);

    // Check shape
    assert_eq!(
        output.size(),
        vec![batch_size as i64, total_num_steps as i64, out_dim as i64]
    );

    // Compare the inner sequences. The output should be the same for each.
    assert_eq!(output.i((.., 0, ..)), output.i((.., 1, ..)));
    assert_eq!(output.i((.., 1..3, ..)), output.i((.., 4..6, ..)));
}

/// Basic structural check of IterativeModule::step
///
/// * Checks that the output size is correct.
/// * Checks that the output state of a step can be used in the next step.
pub fn check_step<M: IterativeModule>(module: &M, in_dim: usize, out_dim: usize) {
    let batch_size: usize = 4;

    let state1 = module.initial_state(batch_size);
    let input1 = Tensor::ones(
        &[batch_size as i64, in_dim as i64],
        (Kind::Float, Device::Cpu),
    );
    let (output1, state2) = module.step(&input1, &state1);
    assert_eq!(output1.size(), vec![batch_size as i64, out_dim as i64]);

    // Make sure the output state can be used as a new input state
    let input2 = -input1;
    let (output2, _) = module.step(&input2, &state2);
    assert_eq!(output2.size(), vec![batch_size as i64, out_dim as i64]);
}
