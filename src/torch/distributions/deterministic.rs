//! Deterministic distribution
use crate::utils::distributions::ArrayDistribution;
use std::convert::TryInto;
use tch::{Device, Kind, Tensor};

/// A deterministic distribution over size-0 vectors.
pub struct DeterministicEmptyVec {
    /// `batch_shape + [0]`
    sampling_shape: Vec<i64>,
}

impl DeterministicEmptyVec {
    pub fn new(mut batch_shape: Vec<i64>) -> Self {
        batch_shape.push(0);
        Self {
            sampling_shape: batch_shape,
        }
    }
}

impl ArrayDistribution<Tensor, Tensor> for DeterministicEmptyVec {
    fn batch_shape(&self) -> Vec<usize> {
        self.sampling_shape
            .split_last()
            .unwrap()
            .1
            .iter()
            .map(|&s| s.try_into().unwrap())
            .collect()
    }

    fn element_shape(&self) -> Vec<usize> {
        vec![0]
    }

    fn sample(&self) -> Tensor {
        Tensor::empty(&self.sampling_shape, (Kind::Int64, Device::Cpu))
    }

    fn log_probs(&self, _elements: &Tensor) -> Tensor {
        Tensor::zeros(
            self.sampling_shape.split_last().unwrap().1,
            (Kind::Float, Device::Cpu),
        )
    }
    fn entropy(&self) -> Tensor {
        Tensor::zeros(
            self.sampling_shape.split_last().unwrap().1,
            (Kind::Float, Device::Cpu),
        )
    }
    fn kl_divergence_from(&self, other: &Self) -> Tensor {
        let self_batch_shape = self.sampling_shape.split_last().unwrap().1;
        let other_batch_shape = other.sampling_shape.split_last().unwrap().1;
        let shape =
            broadcast_shapes(self_batch_shape, other_batch_shape).expect("Mismatched shapes");
        Tensor::zeros(&shape, (Kind::Float, Device::Cpu))
    }
}

fn broadcast_shapes<'a>(mut a: &'a [i64], mut b: &'a [i64]) -> Option<Vec<i64>> {
    // Ensure that b is not longer. It will be prepended with 1s
    if b.len() > a.len() {
        std::mem::swap(&mut a, &mut b);
    }
    let mut broadcasted = Vec::new();
    for (&ai, &bi) in a
        .iter()
        .rev()
        .zip(b.iter().rev().chain(std::iter::repeat(&1)))
    {
        if ai == bi || bi == 1 {
            broadcasted.push(ai);
        } else if ai == 1 {
            broadcasted.push(bi)
        } else {
            return None;
        }
    }
    broadcasted.reverse();
    Some(broadcasted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_shape() {
        let d = DeterministicEmptyVec::new(vec![2, 3]);
        assert_eq!(d.batch_shape(), vec![2, 3]);
    }

    #[test]
    fn element_shape() {
        let d = DeterministicEmptyVec::new(vec![2, 3]);
        assert_eq!(d.element_shape(), vec![0]);
    }

    #[test]
    fn sample() {
        let d = DeterministicEmptyVec::new(vec![2, 3]);
        assert_eq!(
            d.sample(),
            Tensor::empty(&[2, 3, 0], (Kind::Int64, Device::Cpu))
        );
    }

    #[test]
    fn log_probs() {
        let d = DeterministicEmptyVec::new(vec![2, 3]);
        let elements = Tensor::empty(&[2, 3, 0], (Kind::Int64, Device::Cpu));
        assert_eq!(
            d.log_probs(&elements),
            Tensor::zeros(&[2, 3], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn entropies() {
        let d = DeterministicEmptyVec::new(vec![2, 3]);
        assert_eq!(
            d.entropy(),
            Tensor::zeros(&[2, 3], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn kl_divergence() {
        let d = DeterministicEmptyVec::new(vec![2, 3]);
        assert_eq!(
            d.kl_divergence_from(&d),
            Tensor::zeros(&[2, 3], (Kind::Float, Device::Cpu))
        );
    }

    #[test]
    fn kl_divergence_broadcast() {
        let d1 = DeterministicEmptyVec::new(vec![1, 3, 1]);
        let d2 = DeterministicEmptyVec::new(vec![4, 2, 1, 1]);
        assert_eq!(
            d1.kl_divergence_from(&d2),
            Tensor::zeros(&[4, 2, 3, 1], (Kind::Float, Device::Cpu))
        );
    }
}
