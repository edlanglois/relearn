//! Sequence modules test utilities.
use super::{IterativeModule, SequenceModule, StatefulIterativeModule};
use std::iter;
use tch::{self, kind::Kind, nn::Module, Device, IndexOp, Tensor};

/// Basic structural check of [tch::nn::Module::forward] via Tensor::apply
pub fn check_forward<M: Module>(
    module: &M,
    in_dim: usize,
    out_dim: usize,
    batch_shape: &[usize],
    kind: Kind,
) {
    let _no_grad_guard = tch::no_grad_guard();
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
    let _no_grad_guard = tch::no_grad_guard();
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

/// Basic check of SequenceModule::seq_packed
///
/// * Checks that the output size is correct.
/// * Checks that identical inner sequences produce identical output.
pub fn check_seq_packed<M: SequenceModule>(module: &M, in_dim: usize, out_dim: usize) {
    let _no_grad_guard = tch::no_grad_guard();
    // Input consists of 3 sequences: [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3], and [0.1].
    let data = [0.1f32, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4];
    let inputs = Tensor::of_slice(&data)
        .unsqueeze(-1)
        .expand(&[-1, in_dim as i64], false);
    let batch_sizes = Tensor::of_slice(&[3i64, 2, 2, 1]);

    let output = module.seq_packed(&inputs, &batch_sizes);

    // Check shape
    assert_eq!(output.size(), vec![data.len() as i64, out_dim as i64],);

    // Compare the packed sequences.
    // The output should be the same for each since they have the same values.
    let seq_1_indices: &[i64] = &[0, 3, 5, 7];
    let seq_2_indices: &[i64] = &[1, 4, 6];
    let seq_3_indices: &[i64] = &[2];

    assert_eq!(
        output.i((&seq_1_indices[..3], ..)),
        output.i((seq_2_indices, ..))
    );
    assert_eq!(
        output.i((&seq_1_indices[..1], ..)),
        output.i((seq_3_indices, ..))
    );
}

/// Basic structural check of IterativeModule::step
///
/// * Checks that the output size is correct.
/// * Checks that the output state of a step can be used in the next step.
pub fn check_step<M: IterativeModule>(module: &M, in_dim: usize, out_dim: usize) {
    let _no_grad_guard = tch::no_grad_guard();
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

/// Basic check of StatefulIterativeModule::step
///
/// * Checks that the output size is correct.
/// * Checks that multiple steps work.
/// * Checks that reset followed by a step works and matches the first output
pub fn check_stateful_step<M: StatefulIterativeModule>(
    module: &mut M,
    in_dim: usize,
    out_dim: usize,
) {
    let _no_grad_guard = tch::no_grad_guard();
    let input = Tensor::ones(&[in_dim as i64], (Kind::Float, Device::Cpu));
    let output1 = module.step(&input);
    assert_eq!(output1.size(), vec![out_dim as i64]);

    let output2 = module.step(&input);
    assert_eq!(output2.size(), vec![out_dim as i64]);

    module.reset();
    let output3 = module.step(&input);
    assert_eq!(output3.size(), vec![out_dim as i64]);
    assert_eq!(output1, output3);
}
