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

#[cfg(test)]
mod one_hot {
    use super::*;

    #[test]
    fn scalar_i32() {
        assert_eq!(
            one_hot(&Tensor::from(1i64), 4, Kind::Int),
            Tensor::of_slice(&[0, 1, 0, 0])
        );
    }

    #[test]
    fn scalar_i64() {
        assert_eq!(
            one_hot(&Tensor::from(1i64), 4, Kind::Int64),
            Tensor::of_slice(&[0, 1, 0, 0])
        );
    }

    #[test]
    fn scalar_f32() {
        assert_eq!(
            one_hot(&Tensor::from(3i64), 4, Kind::Float),
            Tensor::of_slice(&[0.0, 0.0, 0.0, 1.0f32])
        );
    }

    #[test]
    #[should_panic]
    fn scalar_from_f32_fails() {
        let _ = one_hot(&Tensor::from(1.0f32), 4, Kind::Float);
    }

    #[test]
    #[should_panic]
    fn scalar_index_too_big_fails() {
        let _ = one_hot(&Tensor::from(4i64), 4, Kind::Float);
    }

    #[test]
    #[should_panic]
    fn scalar_index_negative_fails() {
        let _ = one_hot(&Tensor::from(-1i64), 4, Kind::Float);
    }

    #[test]
    fn one_dim() {
        assert_eq!(
            one_hot(&Tensor::of_slice(&[2i64, 1]), 3, Kind::Int),
            Tensor::of_slice(&[0, 0, 1, 0, 1, 0]).view((2, 3))
        );
    }

    #[test]
    fn two_dims() {
        assert_eq!(
            one_hot(&Tensor::of_slice(&[2i64, 1]).view((1, 2)), 3, Kind::Int),
            Tensor::of_slice(&[0, 0, 1, 0, 1, 0]).view((1, 2, 3))
        );
    }
}
