//! Torch utilities.
use std::borrow::Borrow;
use tch::{Kind, TchError, Tensor};

/// Create a one-hot tensor from a tensor of indices,
///
/// The same as [`one_hot`] but returns a result instead of panicking
/// when there is an error.
///
/// # Errors
/// Returns any error raised `tch` or the Torch C++ api.
/// These are generally related to incorrect `Tensor` shapes or types.
pub fn f_one_hot(labels: &Tensor, num_classes: usize, kind: Kind) -> Result<Tensor, TchError> {
    let mut shape = labels.size();
    shape.push(num_classes as i64);
    Tensor::f_zeros(&shape, (kind, labels.device()))?.f_scatter_value_(-1, &labels.unsqueeze(-1), 1)
}

pub fn f_one_hot_out(labels: &Tensor, out: &mut Tensor) -> Result<(), TchError> {
    let _ = out.f_zero_()?;
    let _ = out.f_scatter_value_(-1, &labels.unsqueeze(-1), 1)?;
    Ok(())
}

/// Create a one-hot tensor from a tensor of indices,
///
/// Like [`Tensor::one_hot`] but allows the [`Kind`] to be set.
///
/// [`Kind`]: tch::Kind
///
/// # Args
/// * `labels` - An i64 tensor of class labels in `[0, num_classes)`.
///              Has any shape `[BATCH_SHAPE..]`.
/// * `num_classes` - Total number of classes.
/// * `kind` - The data type of the resulting tensor.
///
/// # Returns
/// A tensor with shape `[BATCH_SHAPE.., num_classes]`
/// equal to 1 at indices `[*idx, labels[*idx]]` and 0 everywhere else.
///
/// # Panics
/// If [`f_one_hot`] fails.
pub fn one_hot(labels: &Tensor, num_classes: usize, kind: Kind) -> Tensor {
    f_one_hot(labels, num_classes, kind).unwrap()
}

/// Write one-hot vectors into the given tensor.
///
/// # Args
/// * `labels` - An i64 tensor of class labels in `[0, num_classes)`.
///              Has any shape `[BATCH_SHAPE..]`.
/// * `out` - Output tensor in which to write the one-hot vectors.
///           Must have shape `[LABELS_SHAPE.., num_classes]`.
///
/// # Panics
/// If [`f_one_hot_out`] fails.
pub fn one_hot_out(labels: &Tensor, out: &mut Tensor) {
    f_one_hot_out(labels, out).unwrap()
}

/// Flatten a set of tensors into a single vector.
///
/// # Errors
/// Returns any error raised `tch` or the Torch C++ api.
/// These are generally related to incorrect `Tensor` shapes or types.
pub fn f_flatten_tensors<I>(tensors: I) -> Result<Tensor, TchError>
where
    I: IntoIterator,
    I::Item: Borrow<Tensor>,
{
    Tensor::f_cat(
        &tensors
            .into_iter()
            .map(|t| t.borrow().f_flatten(0, -1))
            .collect::<Result<Vec<_>, _>>()?,
        0,
    )
}

/// Flatten a set of tensors into a single tensor.
///
/// # Warning
/// Undefined tensors are silently flattened to nothing.
///
/// # Panics
/// If [`f_flatten_tensors`] fails.
pub fn flatten_tensors<I>(tensors: I) -> Tensor
where
    I: IntoIterator,
    I::Item: Borrow<Tensor>,
{
    f_flatten_tensors(tensors).unwrap()
}

/// The number of elements in a Tensor with the given shape.
///
/// # Panics
/// If any dimension has negative size.
fn shape_size(shape: &[i64]) -> i64 {
    assert!(
        shape.iter().all(|&d| d >= 0),
        "Negative dimension in shape {:?}",
        shape
    );
    shape.iter().product()
}

/// Unflatten a vector into a set of tensors with the given shapes.
///
/// # Errors
/// Returns any error raised `tch` or the Torch C++ api.
/// These are generally related to incorrect `Tensor` shapes or types.
///
/// # Panics
/// Panics if any shape has a dimension with negative size.
pub fn f_unflatten_tensors(vector: &Tensor, shapes: &[Vec<i64>]) -> Result<Vec<Tensor>, TchError> {
    let sizes: Vec<_> = shapes.iter().map(|shape| shape_size(shape)).collect();
    vector
        .f_split_with_sizes(&sizes, 0)?
        .iter()
        .zip(shapes)
        .map(|(t, shape)| t.f_reshape(shape))
        .collect()
}

/// Unflatten a vector into a set of tensors with the given shapes.
///
/// # Panics
/// If [`f_unflatten_tensors`] fails.
pub fn unflatten_tensors(vector: &Tensor, shapes: &[Vec<i64>]) -> Vec<Tensor> {
    f_unflatten_tensors(vector, shapes).unwrap()
}

/// Dot product of two flattened tensors.
///
/// # Errors
/// Returns any error raised `tch` or the Torch C++ api.
/// These are generally related to incorrect `Tensor` shapes or types.
pub fn f_flat_dot(a: &Tensor, b: &Tensor) -> Result<Tensor, TchError> {
    a.f_flatten(0, -1)?.f_dot(&b.f_flatten(0, -1)?)
}

/// Dot product of two flattened tensors.
///
/// Equivalently, the sum of the elementwise product of two tensors.
/// The shapes may differ so long as the total number of elements are the same.
///
/// # Panics
/// If [`f_flat_dot`] fails.
pub fn flat_dot(a: &Tensor, b: &Tensor) -> Tensor {
    f_flat_dot(a, b).unwrap()
}

#[cfg(test)]
mod one_hot {
    use super::*;

    #[test]
    fn scalar_i32() {
        assert_eq!(
            one_hot(&Tensor::from(1_i64), 4, Kind::Int),
            Tensor::of_slice(&[0, 1, 0, 0])
        );
    }

    #[test]
    fn scalar_i64() {
        assert_eq!(
            one_hot(&Tensor::from(1_i64), 4, Kind::Int64),
            Tensor::of_slice(&[0, 1, 0, 0])
        );
    }

    #[test]
    fn scalar_f32() {
        assert_eq!(
            one_hot(&Tensor::from(3_i64), 4, Kind::Float),
            Tensor::of_slice(&[0.0, 0.0, 0.0, 1.0_f32])
        );
    }

    #[test]
    #[should_panic]
    fn scalar_from_f32_fails() {
        let _ = one_hot(&Tensor::from(1.0_f32), 4, Kind::Float);
    }

    #[test]
    #[should_panic]
    fn scalar_index_too_big_fails() {
        let _ = one_hot(&Tensor::from(4_i64), 4, Kind::Float);
    }

    #[test]
    #[should_panic]
    fn scalar_index_negative_fails() {
        let _ = one_hot(&Tensor::from(-1_i64), 4, Kind::Float);
    }

    #[test]
    fn one_dim() {
        assert_eq!(
            one_hot(&Tensor::of_slice(&[2_i64, 1]), 3, Kind::Int),
            Tensor::of_slice(&[0, 0, 1, 0, 1, 0]).view((2, 3))
        );
    }

    #[test]
    fn two_dims() {
        assert_eq!(
            one_hot(&Tensor::of_slice(&[2_i64, 1]).view((1, 2)), 3, Kind::Int),
            Tensor::of_slice(&[0, 0, 1, 0, 1, 0]).view((1, 2, 3))
        );
    }
}

#[cfg(test)]
mod one_hot_out {
    use super::*;
    use tch::Device;

    #[test]
    fn scalar_i32() {
        let mut out = Tensor::empty(&[4], (Kind::Int, Device::Cpu));
        one_hot_out(&Tensor::from(1_i64), &mut out);
        assert_eq!(out, Tensor::of_slice(&[0, 1, 0, 0]));
    }

    #[test]
    fn scalar_i64() {
        let mut out = Tensor::empty(&[4], (Kind::Int64, Device::Cpu));
        one_hot_out(&Tensor::from(1_i64), &mut out);
        assert_eq!(out, Tensor::of_slice(&[0, 1, 0, 0]));
    }

    #[test]
    fn scalar_f32() {
        let mut out = Tensor::empty(&[4], (Kind::Float, Device::Cpu));
        one_hot_out(&Tensor::from(3_i64), &mut out);
        assert_eq!(out, Tensor::of_slice(&[0.0, 0.0, 0.0, 1.0_f32]));
    }

    #[test]
    #[should_panic]
    fn scalar_from_f32_fails() {
        let mut out = Tensor::empty(&[4], (Kind::Float, Device::Cpu));
        one_hot_out(&Tensor::from(1.0_f32), &mut out);
    }

    #[test]
    #[should_panic]
    fn scalar_index_too_big_fails() {
        let mut out = Tensor::empty(&[4], (Kind::Float, Device::Cpu));
        one_hot_out(&Tensor::from(4_i64), &mut out);
    }

    #[test]
    #[should_panic]
    fn scalar_index_negative_fails() {
        let mut out = Tensor::empty(&[4], (Kind::Float, Device::Cpu));
        one_hot_out(&Tensor::from(-1_i64), &mut out);
    }

    #[test]
    fn one_dim() {
        let mut out = Tensor::empty(&[2, 3], (Kind::Int, Device::Cpu));
        one_hot_out(&Tensor::of_slice(&[2_i64, 1]), &mut out);
        assert_eq!(out, Tensor::of_slice(&[0, 0, 1, 0, 1, 0]).view((2, 3)));
    }

    #[test]
    fn two_dims() {
        let mut out = Tensor::empty(&[1, 2, 3], (Kind::Int, Device::Cpu));
        one_hot_out(&Tensor::of_slice(&[2_i64, 1]).view((1, 2)), &mut out);
        assert_eq!(out, Tensor::of_slice(&[0, 0, 1, 0, 1, 0]).view((1, 2, 3)));
    }
}

#[cfg(test)]
mod flatten {
    use super::*;

    #[test]
    fn test_flatten_tensors() {
        let a = Tensor::of_slice(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let b = Tensor::of_slice(&[10, 11, 12, 13]).reshape(&[4, 1, 1]);

        let v = flatten_tensors(&[a, b]);
        assert_eq!(v, Tensor::of_slice(&[1, 2, 3, 4, 5, 6, 10, 11, 12, 13]));
    }

    #[test]
    fn test_unflatten_tensors() {
        let v = Tensor::of_slice(&[1, 2, 3, 4, 5, 6, 10, 11, 12, 13]);
        let shapes = [vec![2, 3], vec![4, 1, 1]];
        let ts = unflatten_tensors(&v, &shapes);

        let a = Tensor::of_slice(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
        let b = Tensor::of_slice(&[10, 11, 12, 13]).reshape(&[4, 1, 1]);
        assert_eq!(ts, vec![a, b]);
    }
}

#[cfg(test)]
mod flat_dot {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn test_flat_dot() {
        let a = Tensor::of_slice(&[1, 2, 3, 4]).reshape(&[2, 2]);
        let b = Tensor::of_slice(&[10, 9, 8, 7]).reshape(&[2, 2]);
        let expected = Tensor::scalar_tensor(10 + 9 * 2 + 8 * 3 + 4 * 7, (Kind::Int, Device::Cpu));
        assert_eq!(flat_dot(&a, &b), expected);
    }
}
