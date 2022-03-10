//! Module test utilities.
use super::{BuildModule, FeedForwardModule, IterativeModule, Module, SequenceModule};
use crate::torch::initializers::{Initializer, VarianceScale};
use crate::torch::optimizers::{BuildOptimizer, OnceOptimizer, SgdConfig};
use serde::{de::DeserializeOwned, Serialize};
use smallvec::SmallVec;
use std::fmt::Debug;
use std::iter;
use tch::{self, kind::Kind, Device, IndexOp, Tensor};

/// Basic structural check of [`FeedForwardModule::forward`].
pub fn check_forward<M: FeedForwardModule>(
    module: &M,
    in_dim: usize,
    out_dim: usize,
    batch_shape: &[usize],
    kind: Kind,
) {
    let _no_grad_guard = tch::no_grad_guard();
    let input_shape: Vec<_> = batch_shape
        .iter()
        .chain(iter::once(&in_dim))
        .map(|&d| d as i64)
        .collect();
    let input = Tensor::ones(&input_shape, (kind, Device::Cpu));
    let output = module.forward(&input);
    let mut output_shape = input_shape;
    *output_shape.last_mut().unwrap() = out_dim as i64;
    assert_eq!(output.size(), output_shape);
}

/// Basic check of [`SequenceModule::seq_serial`]
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
    assert_allclose(&output.i((.., 0, ..)), &output.i((.., 1, ..)));
    assert_allclose(&output.i((.., 1..3, ..)), &output.i((.., 4..6, ..)));
}

fn assert_allclose(input: &Tensor, other: &Tensor) {
    assert!(input.allclose(other, 1e-5, 1e-8, false))
}

/// Basic check of [`SequenceModule::seq_packed`]
///
/// * Checks that the output size is correct.
/// * Checks that identical inner sequences produce identical output.
pub fn check_seq_packed<M: SequenceModule>(module: &M, in_dim: usize, out_dim: usize) {
    let _no_grad_guard = tch::no_grad_guard();
    // Input consists of 3 sequences: [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3], and [0.1].
    let data = [0.1_f32, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4];
    let inputs = Tensor::of_slice(&data)
        .unsqueeze(-1)
        .expand(&[-1, in_dim as i64], false);
    let batch_sizes = Tensor::of_slice(&[3_i64, 2, 2, 1]);

    let output = module.seq_packed(&inputs, &batch_sizes);

    // Check shape
    assert_eq!(output.size(), vec![data.len() as i64, out_dim as i64],);

    // Compare the packed sequences.
    // The output should be the same for each since they have the same values.
    let seq_1_indices: &[i64] = &[0, 3, 5, 7];
    let seq_2_indices: &[i64] = &[1, 4, 6];
    let seq_3_indices: &[i64] = &[2];

    assert_allclose(
        &output.i((&seq_1_indices[..3], ..)),
        &output.i((seq_2_indices, ..)),
    );
    assert_allclose(
        &output.i((&seq_1_indices[..1], ..)),
        &output.i((seq_3_indices, ..)),
    );
}

/// Basic structural check of [`IterativeModule::step`]
///
/// * Checks that the output size is correct.
/// * Checks that the output state of a step can be used in the next step.
pub fn check_step<M: IterativeModule>(module: &M, in_dim: usize, out_dim: usize) {
    let _no_grad_guard = tch::no_grad_guard();

    let mut state = module.initial_state();
    let input1 = Tensor::ones(&[in_dim as i64], (Kind::Float, Device::Cpu));
    let output1 = module.step(&mut state, &input1);
    assert_eq!(output1.size(), vec![out_dim as i64]);

    // Make sure the output state can be used as a new input state
    let input2 = -input1;
    let output2 = module.step(&mut state, &input2);
    assert_eq!(output2.size(), vec![out_dim as i64]);
}

/// Check that [`SequenceModule::seq_packed`] output matches [`IterativeModule::step`].
pub fn check_seq_packed_matches_iter_steps<M>(module: &M, in_dim: usize, out_dim: usize)
where
    M: SequenceModule + IterativeModule,
{
    let _no_grad_guard = tch::no_grad_guard();

    let seq_len = 5;
    let num_seqs = 2;
    let input = Tensor::rand(
        &[seq_len, num_seqs, in_dim as i64],
        (Kind::Float, Device::Cpu),
    );

    let packed_input = input.reshape(&[seq_len * num_seqs, in_dim as i64]);
    let batch_sizes = Tensor::full(&[seq_len], num_seqs, (Kind::Int64, Device::Cpu));
    let packed_output = module.seq_packed(&packed_input, &batch_sizes);
    let output = packed_output.reshape(&[seq_len, num_seqs, out_dim as i64]);

    for i in 0..num_seqs {
        let mut state = module.initial_state();
        for j in 0..seq_len {
            let step_output = module.step(&mut state, &input.i((j, i, ..)));
            let expected = output.i((j, i, ..));
            assert!(
                step_output.allclose(&expected, 1e-6, 1e-6, false),
                "seq {i}, step {j}; {step_output:?} != {:?}",
                expected
            );
        }
    }
}

/// Check that gradient descent improves the output of a model.
pub fn check_config_gradient_descent<R, MC>(config: &MC)
where
    MC: BuildModule,
    R: RunModule<MC::Module>,
{
    let in_dim = 2;
    let out_dim = 32; // needs to be large enough to avoid all 0 from ReLU by chance
    let device = Device::Cpu;

    // Initializer for input and target tensors: Unif[-1,1]
    let init = Initializer::Uniform(VarianceScale::Constant(1.0 / 3.0));
    let input = R::new_input(init, in_dim, device);
    let target = init
        .tensor(&R::output_shape(out_dim))
        .device(device)
        .requires_grad(false)
        .build();

    let module = config.build_module(in_dim, out_dim, device);
    let mut optimizer = SgdConfig::default()
        .build_optimizer(module.trainable_variables())
        .unwrap();

    let initial_output = R::run(&module, &input);
    let initial_loss = (&initial_output - &target).square().sum(Kind::Float);

    optimizer
        .backward_step_once(&initial_loss, &mut ())
        .unwrap();

    let final_output = R::run(&module, &input);
    assert_ne!(initial_output, final_output);

    let final_loss = (&final_output - &target).square().sum(Kind::Float);
    let initial_loss_value: f32 = initial_loss.into();
    let final_loss_value: f32 = final_loss.into();
    assert!(final_loss_value < initial_loss_value);
}

/// Try to check that a model can be cloned to a new device.
///
/// Constructs a module on `Cuda` if available and clones to `Cpu`.
/// Ends immediately if `Cuda` is not available.
/// Checks that
/// * the original module runs after cloning
/// * the new module runs after cloning and its output matches the original module
pub fn check_config_clone_to_new_device<R, MC>(config: &MC)
where
    MC: BuildModule,
    R: RunModule<MC::Module>,
{
    let in_dim = 2;
    let out_dim = 3;
    let initial_device = Device::cuda_if_available();
    let target_device = Device::Cpu;

    if initial_device == target_device {
        return;
    }

    let init = Initializer::Constant(1.0);
    let original_input = R::new_input(init, in_dim, initial_device);
    let new_input = R::new_input(init, in_dim, target_device);

    let original_module = config.build_module(in_dim, out_dim, initial_device);

    // Clone to target device
    let new_module = original_module.clone_to_device(target_device);

    // Check that the original still works
    let original_output = R::run(&original_module, &original_input);

    // Check that the new module works with input on the target device
    let new_output = R::run(&new_module, &new_input);

    // Check that the ouputs are equal
    assert_allclose(&original_output.to_device(target_device), &new_output);
}

/// Check use of `clone_to_device` to the same device.
///
/// Constructs a module on `Cpu` and clones to `Cpu`.
///
/// Checks that
/// * the original module runs before cloning
/// * the original module runs after cloning
/// * the new module runs after cloning and its output matches the original module
/// * the modules share memory; they are equal after modifying the variables of one.
pub fn check_config_clone_to_same_device<R, MC>(config: &MC)
where
    MC: BuildModule,
    MC::Module: PartialEq + Debug,
    R: RunModule<MC::Module>,
{
    let in_dim = 2;
    let out_dim = 3;
    let device = Device::Cpu;

    // Initializer for input tensor: Unif[-1,1]
    let init = Initializer::Uniform(VarianceScale::Constant(1.0 / 3.0));
    let input = R::new_input(init, in_dim, device);

    let original_module = config.build_module(in_dim, out_dim, device);

    // Check that the original module runs without crashing
    let _ = R::run(&original_module, &input);

    // Clone to target device
    let new_module = original_module.clone_to_device(device);

    // Check that the original still works
    let original_output = R::run(&original_module, &input);

    // Check that the new module works with input on the target device
    let new_output = R::run(&new_module, &input);

    // Check that the ouputs are equal
    assert_allclose(&original_output, &new_output);

    // Modify the variables of the original module and check that the modules are still equal.
    {
        let _no_grad_guard = tch::no_grad_guard();
        for tensor in original_module.variables() {
            let _ = tensor.shallow_clone().fill_(1);
        }
    }
    assert_eq!(original_module, new_module);
}

/// Check that serializing and deserializing a module matches the original in value and output.
pub fn check_ser_de_matches<R, M>(module: &M, in_dim: usize)
where
    M: Module + Serialize + DeserializeOwned + PartialEq + Debug,
    R: RunModule<M>,
{
    let serialized = serde_cbor::to_vec(module).unwrap();
    let deserialized_module: M = serde_cbor::from_slice(&serialized).unwrap();

    assert_eq!(module, &deserialized_module);

    // Initializer for input and target tensors: Unif[-1,1]
    let init = Initializer::Uniform(VarianceScale::Constant(1.0 / 3.0));
    let input = R::new_input(init, in_dim, Device::Cpu);

    let module_output = R::run(module, &input);
    let deserialized_module_output = R::run(&deserialized_module, &input);
    assert_eq!(module_output, deserialized_module_output);
}

pub trait RunModule<M: ?Sized> {
    /// Model input
    type Input;

    /// Generate an input for the model based on an initializer.
    ///
    /// Makes arbitrary choices about the input structure where not specified
    /// (e.g. batch size, sequence length).
    fn new_input(initializer: Initializer, in_dim: usize, device: Device) -> Self::Input;

    /// The shape of an output tensor when the module is given an input from `new_input`.
    ///
    /// `out_dim` is the model's number of output features.
    fn output_shape(out_dim: usize) -> SmallVec<[usize; 4]>;

    /// Run the module on the given input. Produces a [`Tensor`] containing the output.
    fn run(module: &M, input: &Self::Input) -> Tensor;
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RunForward;
impl RunForward {
    const BATCH_SIZE: usize = 3;
}
impl<M> RunModule<M> for RunForward
where
    M: FeedForwardModule + ?Sized,
{
    type Input = Tensor;

    fn new_input(initializer: Initializer, in_dim: usize, device: Device) -> Self::Input {
        initializer
            .tensor(&[Self::BATCH_SIZE, in_dim])
            .device(device)
            .requires_grad(false)
            .build()
    }

    fn output_shape(out_dim: usize) -> SmallVec<[usize; 4]> {
        [Self::BATCH_SIZE, out_dim].into_iter().collect()
    }

    fn run(module: &M, input: &Self::Input) -> Tensor {
        module.forward(input)
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RunSeqSerial;
impl RunSeqSerial {
    const BATCH_SIZE: usize = 2;
    const SEQ_LENGTHS: [usize; 3] = [4, 3, 1];
}
impl<M> RunModule<M> for RunSeqSerial
where
    M: SequenceModule + ?Sized,
{
    /// (inputs, sequence lengths)
    type Input = (Tensor, [usize; 3]);

    fn new_input(initializer: Initializer, in_dim: usize, device: Device) -> Self::Input {
        let total_seq_len = Self::SEQ_LENGTHS.iter().sum();
        let input = initializer
            .tensor(&[Self::BATCH_SIZE, total_seq_len, in_dim])
            .device(device)
            .requires_grad(false)
            .build();
        (input, Self::SEQ_LENGTHS)
    }

    fn output_shape(out_dim: usize) -> SmallVec<[usize; 4]> {
        let total_seq_len = Self::SEQ_LENGTHS.iter().sum();
        [Self::BATCH_SIZE, total_seq_len, out_dim]
            .into_iter()
            .collect()
    }

    fn run(module: &M, input: &Self::Input) -> Tensor {
        module.seq_serial(&input.0, &input.1)
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RunSeqPacked;
impl RunSeqPacked {
    // Represents sequences of length [4, 3, 1]
    const BATCH_SIZES: [usize; 4] = [3, 2, 2, 1];
}
impl<M> RunModule<M> for RunSeqPacked
where
    M: SequenceModule + ?Sized,
{
    /// (inputs, batch sizes)
    type Input = (Tensor, Tensor);

    fn new_input(initializer: Initializer, in_dim: usize, device: Device) -> Self::Input {
        let total_steps = Self::BATCH_SIZES.iter().sum();
        let input = initializer
            .tensor(&[total_steps, in_dim])
            .device(device)
            .requires_grad(false)
            .build();
        let batch_sizes_i64: [i64; 4] = array_init::array_init(|i| Self::BATCH_SIZES[i] as i64);
        let batch_sizes = Tensor::of_slice(&batch_sizes_i64);
        (input, batch_sizes)
    }

    fn output_shape(out_dim: usize) -> SmallVec<[usize; 4]> {
        let total_steps = Self::BATCH_SIZES.iter().sum();
        [total_steps, out_dim].into_iter().collect()
    }

    fn run(module: &M, input: &Self::Input) -> Tensor {
        module.seq_packed(&input.0, &input.1)
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct RunIterStep;
impl RunIterStep {
    const SEQ_LEN: usize = 5;
}
impl<M> RunModule<M> for RunIterStep
where
    M: IterativeModule + ?Sized,
{
    type Input = Vec<Tensor>;

    fn new_input(initializer: Initializer, in_dim: usize, device: Device) -> Self::Input {
        iter::repeat_with(|| {
            initializer
                .tensor(&[in_dim])
                .device(device)
                .requires_grad(false)
                .build()
        })
        .take(Self::SEQ_LEN)
        .collect()
    }

    fn output_shape(out_dim: usize) -> SmallVec<[usize; 4]> {
        [Self::SEQ_LEN, out_dim].into_iter().collect()
    }

    fn run(module: &M, inputs: &Self::Input) -> Tensor {
        let outputs: Vec<Tensor> = module.iter(inputs).collect();
        Tensor::stack(&outputs, 0)
    }
}
