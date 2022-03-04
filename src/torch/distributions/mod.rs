//! Torch statistical distributions
//!
//! These types are all capable of representing a batch of distributions, with arbitrary batch
//! shape. The return values of statistics methods are tensors with the same batch shape.
mod bernoulli;
mod categorical;
mod deterministic;

pub use bernoulli::Bernoulli;
pub use categorical::Categorical;
pub use deterministic::DeterministicEmptyVec;

use tch::{Kind, Tensor};

/// Clamp float values to be finite
fn clamp_float_finite(x: &Tensor) -> Result<Tensor, Kind> {
    match x.kind() {
        Kind::Float => Ok(x.clamp(f64::from(f32::MIN), f64::from(f32::MAX))),
        Kind::Double => Ok(x.clamp(f64::MIN, f64::MAX)),
        kind => Err(kind),
    }
}

/// Clamp float values to be >= the smallest finite float value. (private helper)
fn clamp_float_min(x: &Tensor) -> Result<Tensor, Kind> {
    match x.kind() {
        Kind::Float => Ok(x.clamp_min(f64::from(f32::MIN))),
        Kind::Double => Ok(x.clamp_min(f64::MIN)),
        kind => Err(kind),
    }
}
