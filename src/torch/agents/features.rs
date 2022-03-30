//! Utilities for calculating step history features.
use crate::simulation::PartialStep;
use crate::spaces::{FeatureSpace, ReprSpace, Space};
use crate::utils::packed::PackedSeqIter;
use crate::utils::torch::ExclusiveTensor;
use lazycell::LazyCell;
use tch::{Device, Tensor};

/// View history features as packed tensors.
pub trait PackedHistoryFeaturesView {
    /// Batch sizes in the packing.
    ///
    /// Note: Batch sizes are always >= 0 but the [tch] API uses i64.
    fn batch_sizes(&self) -> &[i64];

    /// Batch sizes in the packing. A 1D i64 tensor.
    fn batch_sizes_tensor(&self) -> &Tensor;

    /// Packed observation features. A 2D f64 tensor.
    fn observation_features(&self) -> &Tensor;

    /// Packed action values.
    ///
    /// A tensor of any type and shape, apart from the first dimension along which actions are
    /// packed. Appropriate for passing to [`ParameterizedDistributionSpace`] methods.
    ///
    /// [`ParameterizedDistributionSpace`]: crate::spaces::ParameterizedDistributionSpace
    fn actions(&self) -> &Tensor;

    /// Packed rewards. A 1D f32 tensor.
    fn rewards(&self) -> &Tensor;

    /// Packed returns (discounted reward-to-go). A 1D f32 tensor.
    ///
    /// The return is the discounted sum of future rewards
    /// (`return = sum_i { reward_i * discount_factor ** i }`)
    /// starting from each step in the episode to the end of the episode.
    ///
    /// # Warning
    /// In the case of interrupted episodes (`Successor::Terminate`), this incorrectly assumes that
    /// all future rewards are zero.
    // TODO: Allow specifying an estimator for terminated episodes.
    fn returns(&self) -> &Tensor;

    /// Discount factor for calculating returns.
    fn discount_factor(&self) -> f64;

    /// Device on which tensors will be placed.
    fn device(&self) -> Device;
}

/// Packed history features with lazy evaluation and caching.
#[derive(Debug)]
pub struct LazyPackedHistoryFeatures<'a, OS: Space + ?Sized, AS: Space + ?Sized> {
    /// Episodes sorted in monotonic decreasing order of length
    episodes: Vec<&'a [PartialStep<OS::Element, AS::Element>]>,
    observation_space: &'a OS,
    action_space: &'a AS,
    discount_factor: f64,
    device: Device,

    /// Indices of each step of the first episode in packed order, finally total number of steps.
    ///
    /// The range `step_offsets[i] .. step_offsets[i+1]` contains the index (in packed order) of
    /// the `i`-the step of each episode that is long enough.
    step_offsets: Vec<usize>,

    cached_batch_sizes: LazyCell<Vec<i64>>,
    cached_batch_sizes_tensor: LazyCell<Tensor>,
    cached_observation_features: LazyCell<Tensor>,
    cached_actions: LazyCell<Tensor>,
    cached_returns: LazyCell<Tensor>,
    cached_rewards: LazyCell<Tensor>,
}

impl<'a, OS, AS> LazyPackedHistoryFeatures<'a, OS, AS>
where
    OS: Space + ?Sized,
    AS: Space + ?Sized,
{
    pub fn new<I>(
        episodes: I,
        observation_space: &'a OS,
        action_space: &'a AS,
        discount_factor: f64,
        device: Device,
    ) -> Self
    where
        I: IntoIterator<Item = &'a [PartialStep<OS::Element, AS::Element>]>,
    {
        let mut episodes: Vec<_> = episodes.into_iter().collect();
        episodes.sort_unstable_by(|&a, &b| b.len().cmp(&a.len()));

        let step_offsets = step_offsets(&episodes);
        assert_eq!(
            *step_offsets.last().unwrap(),
            episodes.iter().map(|ep| ep.len()).sum::<usize>()
        );

        Self {
            episodes,
            observation_space,
            action_space,
            discount_factor,
            device,
            step_offsets,
            cached_batch_sizes: LazyCell::new(),
            cached_batch_sizes_tensor: LazyCell::new(),
            cached_observation_features: LazyCell::new(),
            cached_actions: LazyCell::new(),
            cached_returns: LazyCell::new(),
            cached_rewards: LazyCell::new(),
        }
    }

    pub fn num_steps(&self) -> usize {
        // Must always have at least 1 element
        *self.step_offsets.last().unwrap()
    }

    pub fn num_episodes(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

impl<'a, OS, AS> PackedHistoryFeaturesView for LazyPackedHistoryFeatures<'a, OS, AS>
where
    OS: FeatureSpace + ?Sized,
    AS: ReprSpace<Tensor> + ?Sized,
{
    fn batch_sizes(&self) -> &[i64] {
        self.cached_batch_sizes.borrow_with(|| {
            self.step_offsets
                .windows(2)
                .map(|w| (w[1] - w[0]) as i64)
                .collect()
        })
    }

    fn batch_sizes_tensor(&self) -> &Tensor {
        // Must stay on the CPU
        self.cached_batch_sizes_tensor
            .borrow_with(|| Tensor::of_slice(self.batch_sizes()))
    }

    fn observation_features(&self) -> &Tensor {
        self.cached_observation_features.borrow_with(|| {
            self.observation_space
                .batch_features::<_, Tensor>(
                    PackedSeqIter::from_sorted(&self.episodes).map(|step| &step.observation),
                )
                .to(self.device)
        })
    }

    fn actions(&self) -> &Tensor {
        self.cached_actions.borrow_with(|| {
            self.action_space
                .batch_repr(PackedSeqIter::from_sorted(&self.episodes).map(|step| &step.action))
                .to(self.device)
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn rewards(&self) -> &Tensor {
        self.cached_rewards.borrow_with(|| {
            Tensor::of_slice(
                &PackedSeqIter::from_sorted(&self.episodes)
                    .map(|step| step.reward as f32)
                    .collect::<Vec<_>>(),
            )
            .to(self.device)
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn returns(&self) -> &Tensor {
        self.cached_returns.borrow_with(|| {
            // Returns must be calculated from the end of the episode
            let mut returns = ExclusiveTensor::zeros(*self.step_offsets.last().unwrap());
            let returns_view = returns.as_slice_mut();
            for (ep_idx, episode) in self.episodes.iter().enumerate() {
                // TODO: Allow specifying an estimated value for non-terminal-end-of-episode.
                let mut return_ = 0.0;
                for (step, i) in episode
                    .iter()
                    .rev()
                    .zip(packed_sequence_indices(&self.step_offsets, ep_idx, episode.len()).rev())
                {
                    return_ *= self.discount_factor;
                    return_ += step.reward;
                    returns_view[i] = return_ as f32;
                }
            }
            returns.into_tensor().to(self.device)
        })
    }

    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }

    fn device(&self) -> Device {
        self.device
    }
}

/// Offset of every step of the first sequence, plus one past the end.
///
/// # Args
/// * `sorted_episode_lengths` - Sequences lengths sorted in monotonic decreasing order.
fn step_offsets<T>(sequences: &[&[T]]) -> Vec<usize> {
    let max_episode_len = sequences.first().map_or(0, |seq| seq.len());
    let mut step_offsets = Vec::with_capacity(max_episode_len + 1);
    let mut batch_size = sequences.len();
    let mut offset = 0;
    step_offsets.push(0);
    for step_idx in 0..max_episode_len {
        // Batch size is the number of episodes that include a step at this idx.
        while sequences[batch_size - 1].len() <= step_idx {
            batch_size -= 1;
        }
        offset += batch_size;
        step_offsets.push(offset);
    }
    step_offsets
}

/// Indices of a sequence in a packed array.
///
/// Indices are used rather than providing references into an array because it is hard or
/// impossible to provide an iterator of mutable references into a mutable array without unsafe
/// code.
fn packed_sequence_indices(
    offsets: &[usize],
    seq_idx: usize,
    sequence_length: usize,
) -> impl Iterator<Item = usize> + DoubleEndedIterator + ExactSizeIterator + Clone + '_ {
    offsets[..=sequence_length].windows(2).map(move |w| {
        let idx = w[0] + seq_idx;
        debug_assert!(idx < w[1]);
        idx
    })
}

#[cfg(test)]
#[allow(clippy::needless_pass_by_value)]
mod lazy_features {
    use super::*;
    use crate::envs::Successor::{Continue, Interrupt, Terminate};
    use crate::spaces::{BooleanSpace, IndexSpace};
    use rstest::{fixture, rstest};

    struct StoredHistory<OS: Space, AS: Space> {
        episodes: Vec<Vec<PartialStep<OS::Element, AS::Element>>>,
        observation_space: OS,
        action_space: AS,
        discount_factor: f64,
        device: Device,
    }

    impl<OS: Space, AS: Space> StoredHistory<OS, AS> {
        fn features(&self) -> LazyPackedHistoryFeatures<OS, AS> {
            LazyPackedHistoryFeatures::new(
                self.episodes.iter().map(AsRef::as_ref),
                &self.observation_space,
                &self.action_space,
                self.discount_factor,
                self.device,
            )
        }
    }

    #[fixture]
    fn history() -> StoredHistory<BooleanSpace, IndexSpace> {
        let episodes = vec![
            vec![
                PartialStep::new(true, 0, 1.0, Continue(())),
                PartialStep::new(true, 1, 1.0, Continue(())),
                PartialStep::new(true, 2, 1.0, Continue(())),
                PartialStep::new(true, 3, 1.0, Continue(())),
            ],
            vec![
                PartialStep::new(false, 10, -1.0, Continue(())),
                PartialStep::new(false, 11, -1.0, Continue(())),
                PartialStep::new(false, 12, 0.0, Continue(())),
                PartialStep::new(false, 13, 0.0, Continue(())),
                PartialStep::new(false, 14, 1.0, Continue(())),
                PartialStep::new(false, 15, 1.0, Terminate),
            ],
            vec![
                PartialStep::new(false, 20, 2.0, Continue(())),
                PartialStep::new(true, 21, 2.0, Continue(())),
                PartialStep::new(false, 22, 2.0, Interrupt(true)),
            ],
            vec![PartialStep::new(true, 30, 3.0, Terminate)],
        ];

        // Packing order (by action)
        // [10, 0, 20, 30,
        //  11, 1, 21,
        //  12, 2, 22,
        //  13, 3,
        //  14,
        //  15]

        StoredHistory {
            episodes,
            observation_space: BooleanSpace::new(),
            action_space: IndexSpace::new(31),
            discount_factor: 0.9,
            device: Device::Cpu,
        }
    }

    #[rstest]
    fn num_steps(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert_eq!(history.features().num_steps(), 14);
    }

    #[rstest]
    fn num_episodes(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert_eq!(history.features().num_episodes(), 4);
    }

    #[rstest]
    fn is_empty(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert!(!history.features().is_empty());
    }

    #[rstest]
    #[allow(clippy::float_cmp)]
    fn discount_factor(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert_eq!(history.features().discount_factor(), 0.9);
    }

    #[rstest]
    fn device(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert_eq!(history.features().device(), Device::Cpu);
    }

    #[rstest]
    fn batch_sizes(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert_eq!(history.features().batch_sizes(), &[4, 3, 3, 2, 1, 1]);
    }

    #[rstest]
    fn batch_sizes_tensor(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert_eq!(
            history.features().batch_sizes_tensor(),
            &Tensor::of_slice(&[4, 3, 3, 2, 1, 1])
        );
    }

    #[rstest]
    fn observation_features(history: StoredHistory<BooleanSpace, IndexSpace>) {
        let features = history.features();
        let actual = features.observation_features();
        let expected = &Tensor::of_slice(&[
            0.0, 1.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, //
            0.0, 1.0, 0.0, //
            0.0, 1.0, //
            0.0, //
            0.0f32,
        ])
        .unsqueeze(-1);
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn actions(history: StoredHistory<BooleanSpace, IndexSpace>) {
        let features = history.features();
        let actual = features.actions();
        let expected = &Tensor::of_slice(&[
            10, 0, 20, 30, //
            11, 1, 21, //
            12, 2, 22, //
            13, 3,  //
            14, //
            15i64,
        ]);
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn rewards(history: StoredHistory<BooleanSpace, IndexSpace>) {
        let features = history.features();
        let actual = features.rewards();
        let expected = &Tensor::of_slice(&[
            -1.0, 1.0, 2.0, 3.0, //
            -1.0, 1.0, 2.0, //
            0.0, 1.0, 2.0, //
            0.0, 1.0, //
            1.0, //
            1.0f32,
        ]);
        assert_eq!(actual, expected);
    }

    #[rstest]
    fn returns(history: StoredHistory<BooleanSpace, IndexSpace>) {
        let features = history.features();
        let actual = features.returns();
        let expected = &Tensor::of_slice(&[
            -0.65341, 3.439, 5.42, 3.0, //
            0.3851, 2.71, 3.8, //
            1.539, 1.9, 2.0, //
            1.71, 1.0, //
            1.9, //
            1.0f32,
        ]);
        // eprintln!("actual   {:?}", Vec::<f32>::from(actual));
        // eprintln!("expected {:?}", Vec::<f32>::from(expected));
        assert!(expected.allclose(actual, 1e-5, 1e-5, false));
    }
}
