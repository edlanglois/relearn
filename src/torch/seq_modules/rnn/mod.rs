mod gru;
mod lstm;

pub use gru::Gru;
pub use lstm::Lstm;

use std::convert::TryInto;
use tch::{nn::Path, Cuda, Device, Tensor};

/// cuDNN RNN mode code
///
/// See https://github.com/pytorch/pytorch/blob/d6909732954ad182d13fa8ab9959502a386e9d3a/torch/csrc/api/src/nn/modules/rnn.cpp#L29
enum CudnnRnnMode {
    LSTM = 2,
    GRU = 3,
}

/// Initialize RNN parameters
fn initialize_rnn_params(
    vs: &Path,
    mode: CudnnRnnMode,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> (Vec<Tensor>, i64, Device) {
    let in_dim: i64 = in_dim.try_into().unwrap();
    let hidden_size: i64 = out_dim.try_into().unwrap();
    let gates_size = match mode {
        CudnnRnnMode::LSTM => 4 * hidden_size,
        CudnnRnnMode::GRU => 3 * hidden_size,
    };

    let mut params = Vec::new();
    params.push(vs.kaiming_uniform("weight_ih", &[gates_size, in_dim]));
    params.push(vs.kaiming_uniform("weight_hh", &[gates_size, hidden_size]));
    if bias {
        params.push(vs.zeros("bias_ih", &[gates_size]));
        params.push(vs.zeros("bias_hh", &[gates_size]));
    }

    let device = vs.device();
    if device.is_cuda() && Cuda::cudnn_is_available() {
        // I assume this must act on the tensors in-place?
        let _ = Tensor::internal_cudnn_rnn_flatten_weight(
            &params,
            params.len() as i64,
            in_dim,
            mode as i64,
            hidden_size,
            0,     // No projections
            1,     // Num layers
            true,  // Batch first
            false, // Not bidirectional
        );
    }
    (params, hidden_size, device)
}
