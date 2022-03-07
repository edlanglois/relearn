//! Tensor initializers
use tch::{nn::Path, Device, Kind, Tensor};

/// Tensor initializers.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Initializer {
    /// Initialize to all zeros
    Zeros,
    /// Uniform distribution with variance scaled by the tensor dimensions.
    Uniform(VarianceScale),
    /// Normal distribution with variance scaled by the tensor dimensions.
    Normal(VarianceScale),
    /// Initialize as an orthogonal matrix.
    Orthogonal,
}

/// Variance scaling mode.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum VarianceScale {
    /// Scale based on a constant.
    ///
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
    fn variance(self, shape: &[i64]) -> f64 {
        let (fan_in, fan_out) = calculate_fan_in_and_fan_out(shape);
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
fn calculate_fan_in_and_fan_out(shape: &[i64]) -> (i64, i64) {
    assert!(
        shape.iter().all(|x| *x >= 0),
        "dimensions must be non-negative"
    );
    // Use feature size of 1 if the dimensions are missing instead of returning an error
    let num_input_fmaps = shape.get(1).cloned().unwrap_or(1);
    let num_output_fmaps = shape.get(0).cloned().unwrap_or(1);
    let receptive_field_size: i64 = shape[2..].iter().product();
    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;
    (fan_in, fan_out)
}

impl Initializer {
    /// Create a new tensor with this initialization.
    ///
    /// # Args
    /// * `vs`    - Namespace and storage in which to create the tensor.
    /// * `name`  - Tensor name.
    /// * `shape` - Tensor shape (tch uses `i64`).
    /// * `gain`  - Scaling factor on the initialized values.
    ///             Can be used to compensate for the scaling effect of activation functions.
    ///             See pytorch's [`calculate_gain`][1] function.
    ///
    /// [1]: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
    ///
    pub fn add_tensor(&self, vs: &Path, name: &str, shape: &[i64], gain: f64) -> Tensor {
        match self {
            Self::Zeros => vs.zeros(name, shape),
            Self::Uniform(scaling) => {
                let lim = gain * (3.0 * scaling.variance(shape)).sqrt();
                vs.uniform(name, shape, -lim, lim)
            }
            Self::Normal(scaling) => {
                let mean = 0.0;
                let stddev = gain * scaling.variance(shape).sqrt();
                vs.randn(name, shape, mean, stddev)
            }
            _ => vs.var_copy(name, &self.init(shape, gain, (Kind::Float, vs.device()))),
        }
    }

    /// Initialize a new tensor
    pub fn init(&self, shape: &[i64], gain: f64, options: (Kind, Device)) -> Tensor {
        match self {
            Self::Zeros => Tensor::zeros(shape, options),
            Self::Uniform(scaling) => {
                let lim = gain * (3.0 * scaling.variance(shape)).sqrt();
                Tensor::empty(shape, options).uniform_(-lim, lim)
            }
            Self::Normal(scaling) => {
                let mean = 0.0;
                let stddev = gain * scaling.variance(shape).sqrt();
                Tensor::empty(shape, options).normal_(mean, stddev)
            }
            Self::Orthogonal => init_orthogonal(shape, gain, options),
        }
    }
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

    let (mut q, r) = flattened.linalg_qr("reduced");
    let d = r.diag(0);
    let ph = d.sign();
    q *= ph;

    if num_rows < num_cols {
        let _ = q.t_();
    }

    if gain != 1.0 {
        q *= gain;
    }
    q.reshape(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_is_orthogonal() {
        let n = 5;
        let options = (Kind::Float, Device::Cpu);
        let a = Initializer::Orthogonal.init(&[n, n], 1.0, options);
        // An orthogonal matrix times its transpose should equal the identity matrix
        assert!(a
            .matmul(&a.tr())
            .allclose(&Tensor::eye(n, options), 1e-4, 1e-4, false));
    }
}
