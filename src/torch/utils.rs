//! Torch utilities.
use std::borrow::Borrow;
use tch::{TchError, Tensor};

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
#[must_use]
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
