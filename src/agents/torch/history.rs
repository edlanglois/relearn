//! History buffer utilities
use super::super::Step;
use crate::spaces::{FeatureSpace, Space};
use tch::{self, Tensor};

/// Packed history feature vectors.
///
/// The `TOTAL_STEPS` dimension of any Tensor of Vector contains all steps from all episodes
/// ordered first by the in-episode step index then by the episode index.
/// If all episodes have the same length then `TOTAL_STEPS = EP_LENGTH * NUM_EPISODES`.
/// In general, episodes may have different lengths.
/// Episodes are always ordered from longest to shortest.
///
/// `TOTAL_STEPS` does not necessarily match the order in which steps were added to the buffer.
///
/// See [PackedSequence][PyTPS] for more information.
/// [PyTPS]: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html
pub struct HistoryFeatures<A> {
    /// Packed observation features.
    ///
    /// An f32 tensor of shape [TOTAL_STEPS, NUM_OBS_FEATURES].
    pub observation_features: Tensor,

    /// Packed step actions.
    ///
    /// A vector of length `TOTAL_STEPS`.
    pub actions: Vec<A>,

    /// Packed step returns.
    ///
    /// A 1D f32 tensor of length `TOTAL_STEPS`.
    pub returns: Tensor,

    /// The batch size of each time step. `TOTAL_STEPS = batch_sizes.sum()`.
    ///
    /// A 1D i64 tensor of length `MAX_EP_LENGTH`.
    pub batch_sizes: Tensor,
}

/// A step history buffer.
pub struct HistoryBuffer<O, A> {
    pub discount_factor: f64,

    steps: Vec<Step<O, A>>,
    episodes: Vec<SubSequence>,
    episode_start_index: usize,
}

impl<O, A> HistoryBuffer<O, A> {
    pub fn new(discount_factor: f64, capacity: Option<usize>) -> Self {
        let steps = match capacity {
            Some(c) => Vec::with_capacity(c),
            None => Vec::new(),
        };
        Self {
            discount_factor,
            steps,
            episodes: Vec::new(),
            episode_start_index: 0,
        }
    }
}

impl<O, A> HistoryBuffer<O, A> {
    /// Number of steps in the buffer.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Number of episodes in the buffer.
    pub fn num_episodes(&self) -> usize {
        self.episodes.len()
    }

    /// Add a step to the buffer.
    pub fn push(&mut self, step: Step<O, A>) {
        let episode_done = step.episode_done;
        self.steps.push(step);
        if episode_done {
            // One past the end of the episode
            let episode_end_index = self.steps.len();
            self.episodes.push(SubSequence {
                start: self.episode_start_index,
                length: episode_end_index - self.episode_start_index,
            });
            self.episode_start_index = episode_end_index;
        }
    }

    /// Process all of the stored history into packed feature tensors and clear the buffer.
    pub fn drain_features<OS>(&mut self, observation_space: &OS) -> HistoryFeatures<A>
    where
        OS: Space<Element = O> + FeatureSpace<Tensor>,
    {
        let _no_grad = tch::no_grad_guard();

        // Sort in descending order of length
        self.episodes
            .sort_by(|a, b| a.length.cmp(&b.length).reverse());

        let observation_features = observation_space.batch_features(
            PackingIndices::new(&self.episodes).map(|i| &self.steps[i].observation),
        );

        // Step returns in the same order as self.steps (initially reversed).
        let mut seq_returns: Vec<_> = self
            .steps
            .iter()
            .rev()
            .scan(0.0, |next_return, step| {
                if step.next_observation.is_none() {
                    // Terminal state
                    *next_return = 0.0
                } else if step.episode_done {
                    // Non-terminal end of episode
                    panic!("Non-terminal end of episode not currently supported");
                }
                *next_return *= self.discount_factor;
                *next_return += step.reward;
                Some(*next_return as f32)
            })
            .collect();
        seq_returns.reverse();
        // Returns in packed order
        let returns: Vec<_> = PackingIndices::new(&self.episodes)
            .map(|i| seq_returns[i])
            .collect();
        let returns = Tensor::of_slice(&returns);

        // PackedStepIter only operates on references so in order to get the actions
        // we convert the steps into Some(step) and then take the step.
        let mut steps: Vec<_> = self.steps.drain(..).map(|s| Some(s)).collect();
        let actions: Vec<_> = PackingIndices::new(&self.episodes)
            .map(|i| steps[i].take().unwrap().action)
            .collect();

        let batch_sizes: Vec<_> = PackedBatchSizes::new(&self.episodes)
            .map(|s| s as i64)
            .collect();
        let batch_sizes = Tensor::of_slice(&batch_sizes);

        self.clear();
        HistoryFeatures {
            observation_features,
            actions,
            returns,
            batch_sizes,
        }
    }

    /// Clears the buffer, removing all stored data.
    pub fn clear(&mut self) {
        self.steps.clear();
        self.episodes.clear();
        self.episode_start_index = 0;
    }
}

/// Description of a subsequence
#[derive(Debug, Clone, Eq, PartialEq)]
struct SubSequence {
    /// Index of the first element
    start: usize,
    /// Length of the subsequence.
    length: usize,
}

/// Index iterator that generates a packed sequence when indexing into a sequential array.
///
/// That is, for an array `seq_data` containing contiguous subsequences sorted in descending order
/// of length as given by `sorted_subsequences`,
/// `PackingIndices::new(&sorted_subsequences).map(|i| seq_data[i])`
/// is a packed sequence.
struct PackingIndices<'a> {
    /// Subsequences sorted in monotonic descreasing order of length.
    sorted_subsequences: &'a [SubSequence],

    /// Current offset within a subsequence.
    offset: usize,
    /// Current subsequence index within a batch.
    subseq_index: usize,
    /// Batch size for the current offset.
    batch_size: usize,
}

impl<'a> PackingIndices<'a> {
    pub fn new(sorted_subsequences: &'a [SubSequence]) -> Self {
        let offset = 0;
        let mut batch_size = sorted_subsequences.len();
        update_offset_batch_size(sorted_subsequences, offset, &mut batch_size);
        Self {
            sorted_subsequences,
            offset,
            subseq_index: 0,
            batch_size,
        }
    }
}

impl<'a> Iterator for PackingIndices<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_size == 0 {
            return None;
        }
        let index = self.sorted_subsequences[self.subseq_index].start + self.offset;
        self.subseq_index += 1;
        if self.subseq_index >= self.batch_size {
            self.subseq_index = 0;
            self.offset += 1;
            update_offset_batch_size(self.sorted_subsequences, self.offset, &mut self.batch_size);
        }
        Some(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.sorted_subsequences.iter().map(|ss| ss.length).sum();
        (size, Some(size))
    }
}

/// The batch size of each subsequence offset when packed.
struct PackedBatchSizes<'a> {
    /// Subsequences sorted in monotonic decreasing order of length.
    sorted_subsequences: &'a [SubSequence],

    /// Current offset within a subsequence
    offset: usize,
    /// Number of subsequences with length > offset
    batch_size: usize,
}

impl<'a> PackedBatchSizes<'a> {
    pub fn new(sorted_subsequences: &'a [SubSequence]) -> Self {
        let offset = 0;
        let mut batch_size = sorted_subsequences.len();
        update_offset_batch_size(sorted_subsequences, offset, &mut batch_size);
        Self {
            sorted_subsequences,
            offset: 0,
            batch_size,
        }
    }
}

impl<'a> Iterator for PackedBatchSizes<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_size == 0 {
            None
        } else {
            let current_batch_size = self.batch_size;
            self.offset += 1;
            update_offset_batch_size(self.sorted_subsequences, self.offset, &mut self.batch_size);
            Some(current_batch_size)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let length = match self.sorted_subsequences.first() {
            Some(ss) => ss.length,
            None => 0,
        };
        (length, Some(length))
    }
}

/// Decrement batch size until it is the batch size of the given offset.
fn update_offset_batch_size(
    sorted_subsequences: &[SubSequence],
    offset: usize,
    batch_size: &mut usize,
) {
    while *batch_size > 0 && sorted_subsequences[*batch_size - 1].length <= offset {
        *batch_size -= 1;
    }
}
