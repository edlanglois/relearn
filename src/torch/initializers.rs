//! Tensor initializers
#![allow(clippy::use_self)] // false positive with serde derives
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use tch::{Device, Kind, Tensor};
use thiserror::Error;

/// Tensor initializers.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum Initializer {
    /// Initialize to all zeros
    Zeros,
    /// Initialize all elements to the given constant value.
    Constant(f64),
    /// Uniform distribution with variance scaled by the tensor dimensions.
    Uniform(VarianceScale),
    /// Normal distribution with variance scaled by the tensor dimensions.
    Normal(VarianceScale),
    /// Initialize as an orthogonal matrix.
    Orthogonal,
}

/// Defaults to `Uniform(FanAvg)` a.k.a. Glorot or Xaviar initialization.
///
/// This samples from `Unif(±√(6 / (fan_in + fan_out)))`.
/// For reference, the typical default initialization used by other libraries are:
/// * PyTorch: `Unif(±√(1 / fan_in))`  (proprotional to `Uniform(FanIn)`)
/// * TensorFlow v1: `Unif(0.05)` (equal to `Uniform(Constant(0.05**2 / 3.0))`)
/// * TensorFlow v2: `Unif(±√(6 / (fan_in + fan_out))` (equal to `Uniform(FanAvg)`)
///
impl Default for Initializer {
    fn default() -> Self {
        // FanIn would sometimes fail the gradient-step-reduces-loss unit tests,
        // presumably because the gradients are too large.
        // FanAvg has not failed that test in my experience.
        Self::Uniform(VarianceScale::FanAvg)
    }
}

/// Variance scaling mode.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum VarianceScale {
    /// The initializer sampling variance is set to the given constant.
    Constant(f64),
    /// Scale based on the number of input features.
    ///
    /// The initializer sampling variance is set to `1 / fan_in`.
    ///
    /// Also known as Kaiming or He initialization.
    FanIn,
    /// Scale based on the number of output features.
    ///
    /// The initializer sampling variance is set to `1 / fan_out`.
    ///
    /// Also known as Kaiming or He initialization (fan-out mode).
    FanOut,
    /// Scale based on the average number of input and output features.
    ///
    /// The initializer sampling variance is set to `2 / (fan_in + fan_out)`.
    ///
    /// Also known as Xavier or Glorot initialization.
    FanAvg,
}

impl Default for VarianceScale {
    fn default() -> Self {
        Self::FanIn
    }
}

impl VarianceScale {
    /// Element sampling variance for the given tensor shape.
    ///
    /// # Args
    /// * `shape`   - Shape of the tensor to create.
    /// * `fan_in`  - Number of input features. Calculated from `shape` if `None`.
    /// * `fan_out` - Number of output features. Calculated from `shape` if `None`.
    fn variance(self, shape: &[usize], fan_in: Option<usize>, fan_out: Option<usize>) -> f64 {
        let (fan_in_calc, fan_out_calc) = calculate_fan_in_and_fan_out(shape);
        let fan_in = fan_in.unwrap_or(fan_in_calc);
        let fan_out = fan_out.unwrap_or(fan_out_calc);
        match self {
            Self::Constant(v) => v,
            Self::FanIn => (fan_in as f64).recip(),
            Self::FanOut => (fan_out as f64).recip(),
            Self::FanAvg => 2.0 / (fan_in as f64 + fan_out as f64),
        }
    }
}

/// Calculate fan in and fan out for a tensor shape.
///
/// Based on the pytorch function [of the same name][1].
///
/// [1]: https://github.com/pytorch/pytorch/blob/f87f753bb997b2da82f7d2a561ccb40ab4f6bd9d/torch/nn/init.py#L284-L300
fn calculate_fan_in_and_fan_out(shape: &[usize]) -> (usize, usize) {
    // Use feature size of 1 if the dimensions are missing instead of returning an error
    let num_input_fmaps = shape.get(1).copied().unwrap_or(1);
    let num_output_fmaps = shape.get(0).copied().unwrap_or(1);
    let receptive_field_size: usize = if shape.len() >= 2 {
        shape[2..].iter().product()
    } else {
        1
    };
    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;
    (fan_in, fan_out)
}

impl Initializer {
    /// Start building a new [`Tensor`] using this initializer.
    ///
    /// See the [`TensorBuilder`] methods for more configuration options.
    #[must_use]
    #[inline]
    pub const fn tensor<'a>(&'a self, shape: &'a [usize]) -> TensorBuilder<'a> {
        TensorBuilder::new(self, shape)
    }
}

/// Builder for initializing a new tensor.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TensorBuilder<'a> {
    initializer: &'a Initializer,
    shape: &'a [usize],
    gain: f64,
    fan_in: Option<usize>,
    fan_out: Option<usize>,
    requires_grad: bool,
    kind: Kind,
    device: Device,
}

impl<'a> TensorBuilder<'a> {
    #[must_use]
    #[inline]
    pub const fn new(initializer: &'a Initializer, shape: &'a [usize]) -> Self {
        Self {
            initializer,
            shape,
            gain: 1.0,
            fan_in: None,
            fan_out: None,
            requires_grad: true,
            kind: Kind::Float,
            device: Device::Cpu,
        }
    }

    /// Build the [`Tensor`].
    pub fn build(&self) -> Tensor {
        let options = (self.kind, self.device);
        let shape_i64: SmallVec<[i64; 8]> =
            self.shape.iter().map(|&d| d.try_into().unwrap()).collect();

        let tensor = match &self.initializer {
            Initializer::Zeros => Tensor::zeros(&shape_i64, options),
            Initializer::Constant(v) => Tensor::full(&shape_i64, *v, options),
            Initializer::Uniform(scaling) => {
                let lim = self.gain
                    * (3.0 * scaling.variance(self.shape, self.fan_in, self.fan_out)).sqrt();
                Tensor::empty(&shape_i64, options).uniform_(-lim, lim)
            }
            Initializer::Normal(scaling) => {
                let mean = 0.0;
                let stddev = self.gain
                    * scaling
                        .variance(self.shape, self.fan_in, self.fan_out)
                        .sqrt();
                Tensor::empty(&shape_i64, options).normal_(mean, stddev)
            }
            Initializer::Orthogonal => init_orthogonal(&shape_i64, self.gain, options),
        };
        tensor.set_requires_grad(self.requires_grad)
    }

    /// Set the gain (scaling factor) on the initialized values.
    ///
    /// Can be used to compensate for the scaling effect of activation functions.
    /// See PyTorch's [`calculate_gain`][1] function for more information.
    ///
    /// [1]: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
    #[must_use]
    #[inline]
    pub const fn gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    /// Override the `fan_in` value (number of input features) calculated from `shape`.
    ///
    /// This can be useful when multiple tensors are initialized separately but act together in
    /// a layer to implement a mapping from a collectively larger number of input features.
    /// For example, a weights tensor and a bias tensors might be initialized with a `fan_in` value
    /// of `weights_input_dim + 1`.
    #[must_use]
    #[inline]
    pub const fn fan_in(mut self, fan_in: usize) -> Self {
        self.fan_in = Some(fan_in);
        self
    }

    /// Override the `fan_out` value (number of output features) calculated from `shape`.
    ///
    /// This can be useful when multiple tensors are initialized separately but their outputs
    /// features are concatenated together in a layer.
    #[must_use]
    #[inline]
    pub const fn fan_out(mut self, fan_out: usize) -> Self {
        self.fan_out = Some(fan_out);
        self
    }

    /// Set whether the tensor requires gradient tracking. Defaults to true.
    #[must_use]
    #[inline]
    pub const fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Set the data type [`Kind`] of the tensor to build. Defaults to 32-bit float.
    ///
    /// Only floating-point and complex kinds are allowed.
    #[inline]
    pub const fn kind(mut self, kind: Kind) -> Result<Self, InitializeTensorError> {
        use Kind::*;
        match kind {
            Half | Float | Double | ComplexHalf | ComplexFloat | ComplexDouble | BFloat16 => {}
            _ => return Err(InitializeTensorError::InvalidKind(kind)),
        }
        self.kind = kind;
        Ok(self)
    }

    /// Set the [`Device`] on which the tensor will be created. Defaults to CPU.
    #[must_use]
    #[inline]
    pub const fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

/// Error initializing a [`Tensor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
pub enum InitializeTensorError {
    #[error("unsupported kind {0:?}; expected a float or complex type")]
    InvalidKind(Kind),
}

// Implementation based on PyTorch, which has the following license
//
// # License
// From PyTorch:
//
// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
//
// From Caffe2:
//
// Copyright (c) 2016-present, Facebook Inc. All rights reserved.
//
// All contributions by Facebook:
// Copyright (c) 2016 Facebook Inc.
//
// All contributions by Google:
// Copyright (c) 2015 Google Inc.
// All rights reserved.
//
// All contributions by Yangqing Jia:
// Copyright (c) 2015 Yangqing Jia
// All rights reserved.
//
// All contributions by Kakao Brain:
// Copyright 2019-2020 Kakao Brain
//
// All contributions from Caffe:
// Copyright(c) 2013, 2014, 2015, the respective contributors
// All rights reserved.
//
// All other contributions:
// Copyright(c) 2015, 2016 the respective contributors
// All rights reserved.
//
// Caffe2 uses a copyright model similar to Caffe: each contributor holds
// copyright over their contributions to Caffe2. The project versioning records
// all such contribution and copyright details. If a contributor wants to further
// mark their specific copyright on a particular contribution, they should
// indicate their copyright solely in the commit message of the change when it is
// committed.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
//    and IDIAP Research Institute nor the names of its contributors may be
//    used to endorse or promote products derived from this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
fn init_orthogonal(shape: &[i64], gain: f64, options: (Kind, Device)) -> Tensor {
    // Reference:
    // <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#orthogonal_>
    assert!(
        shape.len() >= 2,
        "tensor for orthogonal init must be at least 2D",
    );
    let _no_grad = tch::no_grad_guard();

    let num_rows = shape[0];
    let num_cols: i64 = shape[1..].iter().product();
    let mut flattened = Tensor::empty(&[num_rows, num_cols], options).normal_(0.0, 1.0);

    if num_rows < num_cols {
        let _ = flattened.t_();
    }

    let (mut q, r) = Tensor::linalg_qr(&flattened, "reduced");
    let d = r.diag(0);
    let ph = d.sign();
    q *= ph;

    if num_rows < num_cols {
        let _ = q.t_();
    }

    #[allow(clippy::float_cmp)] // Should be exact
    if gain != 1.0 {
        q *= gain;
    }
    q = q.reshape(shape);

    // Copy into another tensor to ensure that the data is in C layout
    let mut out = Tensor::empty(shape, options);
    out.copy_(&q);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros() {
        let a = Initializer::Zeros.tensor(&[5]).build();
        assert_eq!(a, Tensor::zeros(&[5], (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn constant() {
        let a = Initializer::Constant(2.0).tensor(&[5]).build();
        assert_eq!(a, Tensor::full(&[5], 2.0, (Kind::Float, Device::Cpu)));
    }

    #[test]
    fn orthogonal_is_orthogonal() {
        let n = 5;
        let a = Initializer::Orthogonal.tensor(&[n, n]).build();
        // An orthogonal matrix times its transpose should equal the identity matrix
        assert!(a.matmul(&a.tr()).allclose(
            &Tensor::eye(n as i64, (Kind::Float, Device::Cpu)),
            1e-4,
            1e-4,
            false
        ));
    }

    #[test]
    fn shape() {
        let a = Initializer::default().tensor(&[2, 3]).build();
        assert_eq!(a.size(), [2, 3]);
    }

    #[test]
    fn gain() {
        // Unif[-1,1] has a variance of 1/3
        let a = Initializer::Uniform(VarianceScale::Constant(1.0 / 3.0))
            .tensor(&[100])
            .gain(0.1)
            .build();
        let max = f32::from(a.max());
        assert!(max <= 0.1, "{max:?}");
        // 100 random samples from [-0.1, 0.1] should almost certainly have a max > 0.075
        assert!(max >= 0.075, "{max:?}");
    }

    #[test]
    fn fan_in_default() {
        // With default fan_in of 100, max is sqrt(3/100) ~= 0.173
        let a = Initializer::Uniform(VarianceScale::FanIn)
            .tensor(&[1, 100])
            .build();
        let max = f32::from(a.max());
        assert!(max <= 0.174, "{max:?}");
        assert!(max >= 0.173 * 0.75, "{max:?}");
    }

    #[test]
    fn fan_in() {
        // With given fan_in of 1, max is sqrt(3/1) ~ 1.73
        let a = Initializer::Uniform(VarianceScale::FanIn)
            .tensor(&[1, 100])
            .fan_in(1)
            .build();
        let max = f32::from(a.max());
        assert!(max <= 1.74, "{max:?}");
        assert!(max >= 1.73 * 0.75, "{max:?}");
    }

    #[test]
    fn fan_out_default() {
        // With default fan_out of 100, max is sqrt(3/100) ~= 0.173
        let a = Initializer::Uniform(VarianceScale::FanOut)
            .tensor(&[100, 1])
            .build();
        let max = f32::from(a.max());
        assert!(max <= 0.174, "{max:?}");
        assert!(max >= 0.173 * 0.75, "{max:?}");
    }

    #[test]
    fn fan_out() {
        // With given fan_in of 1, max is sqrt(3/1) ~ 1.73
        let a = Initializer::Uniform(VarianceScale::FanOut)
            .tensor(&[100, 1])
            .fan_out(1)
            .build();
        let max = f32::from(a.max());
        assert!(max <= 1.74, "{max:?}");
        assert!(max >= 1.73 * 0.75, "{max:?}");
    }

    #[test]
    fn requires_grad_default() {
        let a = Initializer::default().tensor(&[2]).build();
        assert!(a.requires_grad());
    }

    #[test]
    fn requires_grad_true() {
        let a = Initializer::default()
            .tensor(&[2])
            .requires_grad(true)
            .build();
        assert!(a.requires_grad());
    }

    #[test]
    fn requires_grad_false() {
        let a = Initializer::default()
            .tensor(&[2])
            .requires_grad(false)
            .build();
        assert!(!a.requires_grad());
    }

    #[test]
    fn kind_default_float() {
        let a = Initializer::default().tensor(&[2]).build();
        assert_eq!(a.kind(), Kind::Float);
    }

    #[test]
    fn kind_double() {
        let a = Initializer::default()
            .tensor(&[2])
            .kind(Kind::Double)
            .unwrap()
            .build();
        assert_eq!(a.kind(), Kind::Double);
    }

    #[test]
    fn kind_complex() {
        let a = Initializer::default()
            .tensor(&[2])
            .kind(Kind::ComplexFloat)
            .unwrap()
            .build();
        assert_eq!(a.kind(), Kind::ComplexFloat);
    }

    #[test]
    fn kind_int_error() {
        assert!(Initializer::default().tensor(&[2]).kind(Kind::Int).is_err());
    }

    #[test]
    fn device_default_cpu() {
        let a = Initializer::default().tensor(&[2]).build();
        assert_eq!(a.device(), Device::Cpu);
    }

    #[test]
    fn device_cuda_if_available() {
        let device = Device::cuda_if_available();
        let a = Initializer::default().tensor(&[2]).device(device).build();
        assert_eq!(a.device(), device);
    }
}
