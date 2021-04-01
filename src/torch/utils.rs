//! Torch utilities.
use tch::{kind::Kind, TchError, Tensor};

/// Create a one-hot tensor from a tensor of indices,
///
/// The same as one_hot but returns a result instead of panicking
/// when there is an error.
pub fn f_one_hot(labels: &Tensor, num_classes: usize, kind: Kind) -> Result<Tensor, TchError> {
    let mut shape = labels.size();
    shape.push(num_classes as i64);
    Tensor::f_zeros(&shape, (kind, labels.device()))?.f_scatter1(-1, &labels.unsqueeze(-1), 1)
}

/// Create a one-hot tensor from a tensor of indices,
///
/// Like Tensor::one_hot but allows the Kind to be set.
///
/// # Args
/// * `labels` - An i64 tensor with any shape `[*BATCH_SHAPE]`.
/// * `num_classes` - Total number of classes.
/// * `kind` - The data type of the resulting tensor.
///
/// # Returns
/// A tensor with shape `[*BATCH_SHAPE, num_classes]`
/// equal to 1 at indices `[*idx, labels[*idx]]` and 0 everywhere else.
///
pub fn one_hot(labels: &Tensor, num_classes: usize, kind: Kind) -> Tensor {
    f_one_hot(labels, num_classes, kind).unwrap()
}
