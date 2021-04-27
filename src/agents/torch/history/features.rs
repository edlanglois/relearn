//! Utilities for calculating step history features.
use crate::spaces::FeatureSpace;
use crate::utils::packed::{self, PackedBatchSizes, PackingIndices};
use crate::Step;
use std::ops::Range;
use tch::Tensor;

/// Episode index ranges sorted in decreasing order of episode length.
///
/// Episodes must be packed in decreasing order of length.
pub fn sorted_episode_ranges<I>(ranges: I) -> Vec<Range<usize>>
where
    I: IntoIterator<Item = Range<usize>>,
{
    let mut episode_ranges: Vec<_> = ranges.into_iter().collect();
    episode_ranges.sort_by(|a, b| a.len().cmp(&b.len()).reverse());
    episode_ranges
}

/// Iterator over batch sizes in the packing
pub fn packing_batch_sizes<'a>(
    episode_ranges: &'a [Range<usize>],
) -> PackedBatchSizes<'a, Range<usize>> {
    PackedBatchSizes::from_sorted_ranges(episode_ranges)
}

/// Packed observation features. A 2D f32 tensor.
pub fn packed_observation_features<'a, OS, A>(
    steps: &'a [Step<OS::Element, A>],
    episode_ranges: &[Range<usize>],
    observation_space: &OS,
) -> Tensor
where
    OS: FeatureSpace<Tensor>,
{
    observation_space
        .batch_features(PackingIndices::from_sorted(&episode_ranges).map(|i| &steps[i].observation))
}

/// Packed step rewards. A 1D f32 tensor.
pub fn packed_rewards<'a, S, A>(
    steps: &'a [Step<S, A>],
    episode_ranges: &[Range<usize>],
) -> Tensor {
    Tensor::of_slice(
        &PackingIndices::from_sorted(&episode_ranges)
            .map(|i| steps[i].reward as f32)
            .collect::<Vec<_>>(),
    )
}

/// Packed step returns (discounted rewards-to-go). A 1D f32 tensor.
pub fn packed_returns(rewards: &Tensor, batch_sizes: &[i64], discount_factor: f64) -> Tensor {
    packed::packed_tensor_discounted_cumsum_from_end(rewards, batch_sizes, discount_factor)
}

/// Convert an iterator over steps into packed actions.
///
/// The by-value iterator is necessary because actions are not necessarily copy-able.
pub fn into_packed_actions<I, S, A>(steps: I, episode_ranges: &[Range<usize>]) -> Vec<A>
where
    I: IntoIterator<Item = Step<S, A>>,
{
    // Put into Option so that we can take the action when packing.
    let mut some_actions: Vec<_> = steps.into_iter().map(|step| Some(step.action)).collect();
    PackingIndices::from_sorted(&episode_ranges)
        .map(|i| some_actions[i].take().unwrap())
        .collect()
}
