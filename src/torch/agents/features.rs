//! Utilities for calculating step history features.
use crate::envs::Successor;
use crate::simulation::PartialStep;
use crate::spaces::{FeatureSpace, ReprSpace, Space};
use crate::torch::ExclusiveTensor;
use crate::utils::packed::PackedSeqIter;
use crate::utils::sequence::Sequence;
use ndarray::Axis;
use once_cell::unsync::OnceCell;
use std::iter;
use tch::{Device, IndexOp, Tensor};

/// View history features as packed tensors.
pub trait PackedHistoryFeaturesView {
    /// Batch sizes in the packing.
    ///
    /// Note: Batch sizes are always >= 0 but the [tch] API uses i64.
    fn batch_sizes(&self) -> &[i64];

    /// Batch sizes in the extended packing.
    ///
    /// Note: Batch sizes are always >= 0 but the [tch] API uses i64.
    ///
    /// Equal to `[num_episodes]` followed by `batch_sizes()`.
    fn extended_batch_sizes(&self) -> &[i64];

    /// Batch sizes in the packing. A 1D non-negative i64 tensor.
    fn batch_sizes_tensor(&self) -> &Tensor;

    /// Batch sizes in the extended packing. A 1D non-negative i64 tensor.
    ///
    /// Equal to `[num_episodes]` followed by `batch_sizes()`.
    fn extended_batch_sizes_tensor(&self) -> &Tensor;

    /// Packed observation features. A 2D f64 tensor.
    fn observation_features(&self) -> &Tensor;

    /// Packed extended observation features. Includes interrupted successor observations.
    ///
    /// # Returns
    /// * `extended_observations` - A 2D f64 tensor. Rows are the features of `step.observation`
    ///     for each step in an episode followed by the features of `step.next` on the last step of
    ///     the episode if it is `Step::Interrupt` or zeros otherwise.
    /// * `is_invalid` - A 1D boolean tensor with length equal to the number of rows of
    ///     `extended_observations`. Is `true` where the corresponding row of
    ///     `extended_observations` is invalid (non-interrupted end-of-episode).
    fn extended_observation_features(&self) -> (&Tensor, &Tensor);

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
    /// In the case of interrupted episodes (`Successor::Terminate`),
    /// this incorrectly assumes that all future rewards are zero.
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

    cached_extended_batch_sizes: OnceCell<Vec<i64>>,
    cached_batch_sizes_tensor: OnceCell<Tensor>,
    cached_extended_batch_sizes_tensor: OnceCell<Tensor>,
    cached_observation_features: OnceCell<Tensor>,
    cached_extended_observation_features: OnceCell<(Tensor, Tensor)>,
    cached_actions: OnceCell<Tensor>,
    cached_returns: OnceCell<Tensor>,
    cached_rewards: OnceCell<Tensor>,
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
            episodes.iter().map(Sequence::len).sum::<usize>()
        );

        Self {
            episodes,
            observation_space,
            action_space,
            discount_factor,
            device,
            step_offsets,
            cached_extended_batch_sizes: OnceCell::new(),
            cached_batch_sizes_tensor: OnceCell::new(),
            cached_extended_batch_sizes_tensor: OnceCell::new(),
            cached_observation_features: OnceCell::new(),
            cached_extended_observation_features: OnceCell::new(),
            cached_actions: OnceCell::new(),
            cached_returns: OnceCell::new(),
            cached_rewards: OnceCell::new(),
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
        &self.extended_batch_sizes()[1..]
    }

    fn extended_batch_sizes(&self) -> &[i64] {
        self.cached_extended_batch_sizes.get_or_init(|| {
            iter::once(self.episodes.len() as i64)
                .chain(self.step_offsets.windows(2).map(|w| (w[1] - w[0]) as i64))
                .collect()
        })
    }

    fn batch_sizes_tensor(&self) -> &Tensor {
        // Must stay on the CPU
        self.cached_batch_sizes_tensor
            .get_or_init(|| self.extended_batch_sizes_tensor().i(1..))
    }

    fn extended_batch_sizes_tensor(&self) -> &Tensor {
        // Must stay on the CPU
        self.cached_extended_batch_sizes_tensor
            .get_or_init(|| Tensor::of_slice(self.extended_batch_sizes()))
    }

    fn observation_features(&self) -> &Tensor {
        self.cached_observation_features.get_or_init(|| {
            self.observation_space
                .batch_features::<_, Tensor>(
                    PackedSeqIter::from_sorted(&self.episodes).map(|step| &step.observation),
                )
                .to(self.device)
        })
    }

    fn extended_observation_features(&self) -> (&Tensor, &Tensor) {
        let (extended_observations, is_invalid) =
            self.cached_extended_observation_features.get_or_init(|| {
                let observations = PackedSeqIter::from_sorted(
                    self.episodes
                        .iter()
                        .copied()
                        .map(ExtendedEpisodeObservations::from),
                );
                let num_observations = observations.len();
                let num_features = self.observation_space.num_features();

                let mut extended_observations =
                    ExclusiveTensor::<f32, _>::zeros((num_observations, num_features));
                let mut is_invalid = ExclusiveTensor::<bool, _>::zeros(num_observations);
                {
                    let mut extended_observations = extended_observations.array_view_mut();
                    let mut is_invalid = is_invalid.array_view_mut();
                    for (i, obs) in observations.enumerate() {
                        if let Some(obs) = obs {
                            self.observation_space.features_out(
                                obs,
                                extended_observations
                                    .index_axis_mut(Axis(0), i)
                                    .as_slice_mut()
                                    .unwrap(),
                                true,
                            );
                        } else {
                            is_invalid[i] = true;
                        }
                    }
                }

                (
                    Tensor::from(extended_observations).to(self.device),
                    Tensor::from(is_invalid).to(self.device),
                )
            });
        (extended_observations, is_invalid)
    }

    fn actions(&self) -> &Tensor {
        self.cached_actions.get_or_init(|| {
            self.action_space
                .batch_repr(PackedSeqIter::from_sorted(&self.episodes).map(|step| &step.action))
                .to(self.device)
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn rewards(&self) -> &Tensor {
        self.cached_rewards.get_or_init(|| {
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
        self.cached_returns.get_or_init(|| {
            // Returns must be calculated from the end of the episode
            let mut returns = ExclusiveTensor::zeros(*self.step_offsets.last().unwrap());
            let returns_view = returns.as_slice_mut();
            for (ep_idx, episode) in self.episodes.iter().enumerate() {
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
    let max_episode_len = sequences.first().map_or(0, Sequence::len);
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

/// View an episode as a `Sequence` of observations: one per step followed by the final successor.
///
/// All items are `Some` except possibly the final successor observation, which is `None` for
/// `Successor::Terminate` or empty episodes.
struct ExtendedEpisodeObservations<'a, O, A> {
    episode: &'a [PartialStep<O, A>],
}

impl<'a, O, A> From<&'a [PartialStep<O, A>]> for ExtendedEpisodeObservations<'a, O, A> {
    fn from(episode: &'a [PartialStep<O, A>]) -> Self {
        Self { episode }
    }
}

impl<'a, O, A> Sequence for ExtendedEpisodeObservations<'a, O, A> {
    type Item = Option<&'a O>;
    fn len(&self) -> usize {
        // Each step plus the final successor.
        self.episode.len() + 1
    }
    fn is_empty(&self) -> bool {
        false
    }
    fn get(&self, idx: usize) -> Option<Self::Item> {
        let ep_len = self.episode.len();
        if idx < ep_len {
            // Within episode
            Some(Some(&self.episode[idx].observation))
        } else if idx == 0 {
            // One past the end of an empty episode
            Some(None)
        } else if idx == ep_len {
            // One past the end of a non-empty episode.
            // Return step.next if the episode was interrupted, otherwise None.
            Some(match &self.episode[idx - 1].next {
                Successor::Interrupt(obs) => Some(obs),
                _ => None,
            })
        } else {
            // More than one past the end of the episode.
            None
        }
    }
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
