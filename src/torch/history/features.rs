//! Utilities for calculating step history features.
use crate::spaces::{FeatureSpace, Space};
use crate::utils::packed::{self, PackedBatchSizes, PackingIndices};
use crate::Step;
use lazycell::LazyCell;
use std::ops::Range;
use tch::Tensor;

/// View packed history features
pub trait PackedHistoryFeaturesView {
    /// Discount factor for calculating returns.
    fn discount_factor(&self) -> f64;

    /// Episode index ranges sorted in decreasing order of episode length.
    fn episode_ranges(&self) -> &[Range<usize>];

    /// Batch sizes in the packing.
    ///
    /// Note: Batch sizes are always >= 0 but the [tch] API uses i64.
    fn batch_sizes(&self) -> &[i64];

    /// Batch sizes in the packing. A 1D i64 tensor.
    fn batch_sizes_tensor(&self) -> &Tensor;

    /// Packed observation features. A 2D f64 tensor.
    fn observations(&self) -> &Tensor;

    /// Packed returns (discounted reward-to-go). A 1D f32 tensor.
    fn returns(&self) -> &Tensor;

    /// Packed rewards. A 1D f32 tensor.
    fn rewards(&self) -> &Tensor;
}

/// Packed history features with lazy evaluation and caching.
pub struct LazyPackedHistoryFeatures<'a, OS: Space, A> {
    steps: &'a [Step<OS::Element, A>],
    episode_ranges: Vec<Range<usize>>,
    observation_space: &'a OS,
    discount_factor: f64,

    cached_batch_sizes: LazyCell<Vec<i64>>,
    cached_batch_sizes_tensor: LazyCell<Tensor>,
    cached_observation_features: LazyCell<Tensor>,
    cached_returns: LazyCell<Tensor>,
    cached_rewards: LazyCell<Tensor>,
}

impl<'a, OS: Space, A> LazyPackedHistoryFeatures<'a, OS, A> {
    pub fn new<I>(
        steps: &'a [Step<OS::Element, A>],
        episode_ranges: I,
        observation_space: &'a OS,
        discount_factor: f64,
    ) -> Self
    where
        I: IntoIterator<Item = Range<usize>>,
    {
        let episode_ranges = sorted_episode_ranges(episode_ranges);
        Self {
            steps,
            episode_ranges,
            observation_space,
            discount_factor,
            cached_batch_sizes: LazyCell::new(),
            cached_batch_sizes_tensor: LazyCell::new(),
            cached_observation_features: LazyCell::new(),
            cached_returns: LazyCell::new(),
            cached_rewards: LazyCell::new(),
        }
    }

    /// Finalize this into a structure of the current values in cache.
    pub fn finalize(self) -> PackedHistoryFeatures {
        PackedHistoryFeatures {
            discount_factor: self.discount_factor,
            episode_ranges: self.episode_ranges,
            batch_sizes: self.cached_batch_sizes.into_inner(),
            batch_sizes_tensor: self.cached_batch_sizes_tensor.into_inner(),
            observation_features: self.cached_observation_features.into_inner(),
            returns: self.cached_returns.into_inner(),
            rewards: self.cached_rewards.into_inner(),
        }
    }
}

impl<'a, OS, A> PackedHistoryFeaturesView for LazyPackedHistoryFeatures<'a, OS, A>
where
    OS: FeatureSpace<Tensor>,
{
    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }

    fn episode_ranges(&self) -> &[Range<usize>] {
        &self.episode_ranges
    }

    fn batch_sizes(&self) -> &[i64] {
        self.cached_batch_sizes.borrow_with(|| {
            packing_batch_sizes(&self.episode_ranges)
                .map(|x| x as i64)
                .collect()
        })
    }

    fn batch_sizes_tensor(&self) -> &Tensor {
        self.cached_batch_sizes_tensor
            .borrow_with(|| Tensor::of_slice(self.batch_sizes()))
    }

    fn observations(&self) -> &Tensor {
        self.cached_observation_features.borrow_with(|| {
            packed_observation_features(&self.steps, &self.episode_ranges, self.observation_space)
        })
    }

    fn returns(&self) -> &Tensor {
        self.cached_returns.borrow_with(|| {
            packed_returns(self.rewards(), self.batch_sizes(), self.discount_factor)
        })
    }

    fn rewards(&self) -> &Tensor {
        self.cached_rewards
            .borrow_with(|| packed_rewards(&self.steps, &self.episode_ranges))
    }
}

/// Packed history features.
///
/// # Panics
/// The [PackedHistoryFeaturesView] this provides will panic
/// if the requested features is not available.
pub struct PackedHistoryFeatures {
    pub discount_factor: f64,
    pub episode_ranges: Vec<Range<usize>>,
    pub batch_sizes: Option<Vec<i64>>,
    pub batch_sizes_tensor: Option<Tensor>,
    pub observation_features: Option<Tensor>,
    pub returns: Option<Tensor>,
    pub rewards: Option<Tensor>,
}

impl PackedHistoryFeaturesView for PackedHistoryFeatures {
    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }

    fn episode_ranges(&self) -> &[Range<usize>] {
        &self.episode_ranges
    }

    fn batch_sizes(&self) -> &[i64] {
        self.batch_sizes
            .as_ref()
            .expect("batch_sizes has not been evaluated")
    }

    fn batch_sizes_tensor(&self) -> &Tensor {
        self.batch_sizes_tensor
            .as_ref()
            .expect("batch_sizes has not been evaluated")
    }

    fn observations(&self) -> &Tensor {
        self.observation_features
            .as_ref()
            .expect("observation_features has not been evaluated")
    }

    fn returns(&self) -> &Tensor {
        self.returns
            .as_ref()
            .expect("returns has not been evaluated")
    }

    /// Packed rewards. A 1D f32 tensor.
    fn rewards(&self) -> &Tensor {
        self.rewards
            .as_ref()
            .expect("rewards has not been evaluated")
    }
}

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
pub fn packed_observation_features<OS, A>(
    steps: &[Step<OS::Element, A>],
    episode_ranges: &[Range<usize>],
    observation_space: &OS,
) -> Tensor
where
    OS: FeatureSpace<Tensor>,
{
    let _no_grad = tch::no_grad_guard();
    observation_space
        .batch_features(PackingIndices::from_sorted(&episode_ranges).map(|i| &steps[i].observation))
}

/// Packed step rewards. A 1D f32 tensor.
pub fn packed_rewards<S, A>(steps: &[Step<S, A>], episode_ranges: &[Range<usize>]) -> Tensor {
    let _no_grad = tch::no_grad_guard();
    Tensor::of_slice(
        &PackingIndices::from_sorted(&episode_ranges)
            .map(|i| steps[i].reward as f32)
            .collect::<Vec<_>>(),
    )
}

/// Packed step returns (discounted rewards-to-go). A 1D f32 tensor.
pub fn packed_returns(rewards: &Tensor, batch_sizes: &[i64], discount_factor: f64) -> Tensor {
    let _no_grad = tch::no_grad_guard();
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