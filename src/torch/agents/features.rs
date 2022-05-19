//! Utilities for calculating step history features.
use crate::envs::Successor;
use crate::simulation::PartialStep;
use crate::spaces::{FeatureSpace, ReprSpace, Space};
use crate::torch::packed::{PackedSeqIter, PackedStructure, PackedTensor};
use crate::torch::ExclusiveTensor;
use crate::utils::sequence::Sequence;
use ndarray::Axis;
use once_cell::unsync::OnceCell;
use std::cmp::Reverse;
use tch::{Device, Tensor};

/// View history features as packed tensors.
///
/// Floating-point tensors are `f32`.
pub trait PackedHistoryFeaturesView {
    /// Packed observation features. A 2D f64 tensor.
    fn observation_features(&self) -> &PackedTensor;

    /// Packed extended observation features. Includes interrupted successor observations.
    ///
    /// # Returns
    /// * `extended_observations` - A 2D f64 tensor. Rows are the features of `step.observation`
    ///     for each step in an episode followed by the features of `step.next` on the last step of
    ///     the episode if it is `Step::Interrupt` or zeros otherwise.
    /// * `is_invalid` - A 1D boolean tensor with length equal to the number of rows of
    ///     `extended_observations`. Is `true` where the corresponding row of
    ///     `extended_observations` is invalid (non-interrupted end-of-episode).
    fn extended_observation_features(&self) -> (&PackedTensor, &PackedTensor);

    /// Packed action values.
    ///
    /// A tensor of any type and shape, apart from the first dimension along which actions are
    /// packed. Appropriate for passing to [`ParameterizedDistributionSpace`] methods.
    ///
    /// [`ParameterizedDistributionSpace`]: crate::spaces::ParameterizedDistributionSpace
    fn actions(&self) -> &PackedTensor;

    /// Packed rewards. A 1D f32 tensor.
    fn rewards(&self) -> &PackedTensor;

    /// Environment discount factor.
    ///
    /// This is included for convenience as it is often needed
    /// when calculating value functions of these features.
    fn discount_factor(&self) -> f64;

    /// Device on which tensors will be placed.
    fn device(&self) -> Device;
}

/// Packed history features with lazy evaluation and caching.
#[derive(Debug)]
pub struct LazyPackedHistoryFeatures<'a, OS: Space + ?Sized, AS: Space + ?Sized, E> {
    /// Episodes sorted in monotonic decreasing order of length
    episodes: Vec<E>,
    observation_space: &'a OS,
    action_space: &'a AS,
    discount_factor: f64,
    device: Device,

    /// Structure representing sequences that are 1 longer than each episode
    extended_structure: PackedStructure,

    cached_observation_features: OnceCell<PackedTensor>,
    cached_extended_observation_features: OnceCell<(PackedTensor, PackedTensor)>,
    cached_actions: OnceCell<PackedTensor>,
    cached_rewards: OnceCell<PackedTensor>,
}

impl<'a, OS, AS, E> LazyPackedHistoryFeatures<'a, OS, AS, E>
where
    OS: Space + ?Sized,
    AS: Space + ?Sized,
    E: Sequence,
{
    pub fn new<I>(
        episodes: I,
        observation_space: &'a OS,
        action_space: &'a AS,
        discount_factor: f64,
        device: Device,
    ) -> Self
    where
        I: IntoIterator<Item = E>,
    {
        let mut episodes: Vec<_> = episodes.into_iter().collect();
        episodes.sort_unstable_by_key(|ep| Reverse(ep.len()));

        let extended_structure =
            PackedStructure::from_sorted_sequence_lengths(episodes.iter().map(|ep| ep.len() + 1))
                .unwrap();

        Self {
            episodes,
            observation_space,
            action_space,
            discount_factor,
            device,
            extended_structure,
            cached_observation_features: OnceCell::new(),
            cached_extended_observation_features: OnceCell::new(),
            cached_actions: OnceCell::new(),
            cached_rewards: OnceCell::new(),
        }
    }

    pub fn num_steps(&self) -> usize {
        self.extended_structure.len() - self.episodes.len()
    }

    pub fn num_episodes(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Regular non-extended structure
    fn structure(&self) -> PackedStructure {
        self.extended_structure.clone().trim(1)
    }
}

impl<'a, OS, AS, E> PackedHistoryFeaturesView for LazyPackedHistoryFeatures<'a, OS, AS, E>
where
    OS: FeatureSpace + ?Sized,
    AS: ReprSpace<Tensor> + ?Sized,
    // Like &'a [PartialStep<O, A>]
    E: Sequence<Item = &'a PartialStep<OS::Element, AS::Element>>
        + IntoIterator<Item = &'a PartialStep<OS::Element, AS::Element>>
        + Copy,
    E::IntoIter: DoubleEndedIterator,
{
    fn observation_features(&self) -> &PackedTensor {
        self.cached_observation_features.get_or_init(|| {
            let tensor = self
                .observation_space
                .batch_features::<_, Tensor>(
                    PackedSeqIter::from_sorted(&self.episodes).map(|step| &step.observation),
                )
                .to(self.device);
            PackedTensor::from_parts(tensor, self.structure())
        })
    }

    fn extended_observation_features(&self) -> (&PackedTensor, &PackedTensor) {
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

                let packed_extended_observations = PackedTensor::from_parts(
                    extended_observations.into_tensor().to(self.device),
                    self.extended_structure.clone(),
                );
                let packed_is_invalid = PackedTensor::from_parts(
                    is_invalid.into_tensor().to(self.device),
                    self.extended_structure.clone(),
                );

                (packed_extended_observations, packed_is_invalid)
            });
        (extended_observations, is_invalid)
    }

    fn actions(&self) -> &PackedTensor {
        self.cached_actions.get_or_init(|| {
            let tensor = self
                .action_space
                .batch_repr(PackedSeqIter::from_sorted(&self.episodes).map(|step| &step.action))
                .to(self.device);
            PackedTensor::from_parts(tensor, self.structure())
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn rewards(&self) -> &PackedTensor {
        self.cached_rewards.get_or_init(|| {
            let tensor = Tensor::of_slice(
                &PackedSeqIter::from_sorted(&self.episodes)
                    .map(|step| step.reward as f32)
                    .collect::<Vec<_>>(),
            )
            .to(self.device);
            PackedTensor::from_parts(tensor, self.structure())
        })
    }

    fn discount_factor(&self) -> f64 {
        self.discount_factor
    }

    fn device(&self) -> Device {
        self.device
    }
}

/// View an episode as a `Sequence` of observations: one per step followed by the final successor.
///
/// All items are `Some` except possibly the final successor observation, which is `None` for
/// `Successor::Terminate` or empty episodes.
struct ExtendedEpisodeObservations<E> {
    episode: E,
}

impl<E> From<E> for ExtendedEpisodeObservations<E> {
    fn from(episode: E) -> Self {
        Self { episode }
    }
}

impl<'a, E, O, A> Sequence for ExtendedEpisodeObservations<E>
where
    E: Sequence<Item = &'a PartialStep<O, A>>,
    O: 'a,
    A: 'a,
{
    type Item = Option<&'a O>;
    fn len(&self) -> usize {
        // Each step plus the final successor.
        self.episode.len() + 1
    }
    fn is_empty(&self) -> bool {
        false
    }
    fn get(&self, idx: usize) -> Option<Self::Item> {
        match self.episode.get(idx) {
            // Within episode
            Some(step) => Some(Some(&step.observation)),
            // One past the end of an empty episode
            None if idx == 0 => Some(None),
            // One past the end of a non-empty episode
            None if idx == self.episode.len() => {
                // Return step.next if the episode was interrupted, otherwise None
                match &self.episode.get(idx - 1).unwrap().next {
                    Successor::Interrupt(obs) => Some(Some(obs)),
                    _ => Some(None),
                }
            }
            // More than one past the end of the episode
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) mod tests {
    use super::*;
    use crate::envs::Successor::{Continue, Interrupt, Terminate};
    use crate::spaces::{BooleanSpace, IndexSpace};
    use rstest::{fixture, rstest};

    pub struct StoredHistory<OS: Space, AS: Space> {
        episodes: Vec<Vec<PartialStep<OS::Element, AS::Element>>>,
        observation_space: OS,
        action_space: AS,
        discount_factor: f64,
        device: Device,
    }

    impl<OS: Space, AS: Space> StoredHistory<OS, AS> {
        #[allow(clippy::type_complexity)]
        pub fn features(
            &self,
        ) -> LazyPackedHistoryFeatures<OS, AS, &[PartialStep<OS::Element, AS::Element>]> {
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
    pub fn history() -> StoredHistory<BooleanSpace, IndexSpace> {
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
        assert_eq!(actual.tensor(), expected);
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
        assert_eq!(actual.tensor(), expected);
    }

    #[rstest]
    fn actions_batch_sizes_tensor(history: StoredHistory<BooleanSpace, IndexSpace>) {
        assert_eq!(
            history.features().actions().batch_sizes_tensor(),
            Tensor::of_slice(&[4, 3, 3, 2, 1, 1])
        );
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
        assert_eq!(actual.tensor(), expected);
    }
}
